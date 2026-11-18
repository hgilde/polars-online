"""What `.github/workflows/vm-checks.yml` runs, so the workflow stays a list of steps.

The x86-64 VMs time wall-clock; building the release wheels they time, and
checking that a branch's outputs match its base's, is work a GitHub runner
does as well. The workflow is dispatched for one branch and one base:

    python scripts/vm_checks.py plan --repo branch --base BASE --variants default,aligned
    python scripts/vm_checks.py wheels --repo branch --variants default,aligned --out dist
    python scripts/vm_checks.py digests --out digests
    python scripts/vm_checks.py tests --repo branch --package online-core --filter solve:: \
        --qemu-cpu Nehalem

This script, `compare_outputs.py` and the build-prep action come from the
commit the workflow runs from; the branch under test, which usually predates
them, is a second checkout (``--repo``, ``branch/``) that is built, merge-based
and tested from its own tree. ``digests`` needs no ``--repo``: it compares two
wheels.

``plan`` resolves the branch head (the ``--repo`` checkout's ``HEAD``) and its base, by
default the merge-base with ``origin/main``, and writes both shas, short, to
``$GITHUB_OUTPUT`` for the workflow's artifact names. ``wheels`` makes a
release wheel of each requested layout for the head and for the base, one
after the other: the base's from the artifact an earlier run uploaded for that
sha and layout when there is one, built otherwise. ``digests`` installs the
two default wheels into fresh environments and runs
`scripts/compare_outputs.py` on them over its cases. ``tests`` runs a package's
tests that match a filter, ignored ones included, natively on the runner
(x86-64 with AVX2) and, given a CPU, again under ``qemu-x86_64 -cpu`` that CPU:
the same test binaries, so a code path the runner's CPU never takes (the
fallback of a machine without AVX2 and FMA) runs too. A filter that matches no
test fails, where ``cargo test`` would exit 0.

The layouts, each a different code layout of the same source:

``default``  as the release wheel is built;
``aligned``  every function on a 64-byte boundary
             (``-C llvm-args=-align-all-functions=6``);
``v3``       the x86-64-v3 instruction set (``-C target-cpu=x86-64-v3``).

A wheel is named ``wheel-<sha8>-<layout>``, the artifact a VM downloads; the
file inside is built natively on the runner (``--compatibility linux``), so it
needs the runner's glibc or newer. Standard library only, plus ``gh``, ``uv``
and ``git`` on the PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Each layout's rustflags, beside the linker flag every build carries.
LAYOUTS = {
    "default": "",
    "aligned": "-C llvm-args=-align-all-functions=6",
    "v3": "-C target-cpu=x86-64-v3",
}
#: lld, as `.github/actions/linux-build-prep` points cargo at it: the base's
#: checkout has no `.cargo/config.toml` of the job's, and `RUSTFLAGS` replaces
#: the config's rustflags anyway, so every build names the linker itself.
LINKER = "-C link-arg=-fuse-ld=lld"
ROLES = ("branch", "base")


def short(sha: str) -> str:
    return sha[:8]


def wheel_name(sha: str, layout: str) -> str:
    """The artifact a wheel of ``sha`` in ``layout`` is uploaded as."""
    return f"wheel-{short(sha)}-{layout}"


def parse_layouts(text: str, digests: bool = False) -> list[str]:
    """The layouts a ``variants`` input names, in the order given, no repeats.
    ``digests`` needs the default layout's wheels, so it is added when missing."""
    out: list[str] = []
    for word in text.replace(",", " ").split():
        if word not in LAYOUTS:
            raise ValueError(f"unknown variant {word!r}: one of {', '.join(LAYOUTS)}")
        if word not in out:
            out.append(word)
    if digests and "default" not in out:
        out.insert(0, "default")
    if not out:
        raise ValueError("no variant requested")
    return out


def rustflags(layout: str) -> str:
    return f"{LAYOUTS[layout]} {LINKER}".strip()


# --- the commits ------------------------------------------------------------------


def _run(cmd: list[str], **kw) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, encoding="utf-8", **kw
    ).stdout


def git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo).strip()


def resolve_commit(repo: Path, rev: str) -> str:
    """``rev`` (a sha, a branch or a tag) as a full sha, a remote branch's
    when the checkout holds no local one."""
    for candidate in (rev, f"origin/{rev}"):
        try:
            return git(repo, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        except subprocess.CalledProcessError:
            continue
    raise ValueError(f"{rev!r} names no commit in this checkout")


def plan(repo: Path, base: str = "") -> tuple[str, str]:
    """The head's full sha and its base's: ``base`` as given, else the
    merge-base of the head with ``origin/main``."""
    head = git(repo, "rev-parse", "HEAD")
    if base:
        return head, resolve_commit(repo, base)
    return head, git(repo, "merge-base", head, "origin/main")


def write_output(pairs: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.writelines(f"{k}={v}\n" for k, v in pairs.items())


# --- the artifacts an earlier run kept -------------------------------------------------


def _gh(*args: str) -> str:
    return _run(["gh", *args])


def earlier_run(name: str, repository: str) -> str | None:
    """The run that uploaded the newest unexpired artifact called ``name``,
    or None."""
    listing = json.loads(
        _gh("api", f"repos/{repository}/actions/artifacts?name={name}&per_page=100")
    )
    live = [a for a in listing.get("artifacts", []) if a["name"] == name and not a["expired"]]
    if not live:
        return None
    newest = max(live, key=lambda a: a["created_at"])
    return str(newest["workflow_run"]["id"])


def fetch(name: str, run: str, repository: str, into: Path) -> Path:
    """The wheel of artifact ``name`` from ``run``, in ``into``."""
    into.mkdir(parents=True, exist_ok=True)
    _gh("run", "download", run, "--repo", repository, "--name", name, "--dir", str(into))
    return one_wheel(into)


def one_wheel(folder: Path) -> Path:
    found = sorted(folder.glob("*.whl"))
    if len(found) != 1:
        raise RuntimeError(f"{folder} holds {len(found)} wheels, not one")
    return found[0]


# --- building ----------------------------------------------------------------------------


def maturin_requirement(repo: Path) -> str:
    """The maturin the package's own build asks for, as ``pyproject.toml`` writes it."""
    doc = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    (req,) = (r for r in doc["build-system"]["requires"] if r.startswith("maturin"))
    return req


def build_command(repo: Path, out: Path, interpreter: str) -> list[str]:
    return [
        "uvx", "--from", maturin_requirement(repo), "maturin", "build",
        "--release", "--locked", "--compatibility", "linux",
        "--interpreter", interpreter,
        "--out", str(out), "-m", "crates/online-py/Cargo.toml",
    ]  # fmt: skip


def build_wheel(repo: Path, layout: str, target_dir: Path, out: Path) -> Path:
    """A release wheel of the checkout at ``repo`` in ``layout``, into ``out``."""
    out = out.resolve()  # the build runs in ``repo``, a relative ``out`` would move there
    out.mkdir(parents=True, exist_ok=True)
    interpreter = _run(["uv", "python", "find", "3.12"]).strip()
    env = {
        **os.environ,
        "RUSTFLAGS": rustflags(layout),
        "CARGO_TARGET_DIR": str(target_dir),
        "CARGO_INCREMENTAL": "0",
    }
    subprocess.run(build_command(repo, out, interpreter), cwd=repo, env=env, check=True)
    return one_wheel(out)


def make_wheels(
    repo: Path, layouts: list[str], head: str, base: str, out: Path, fetched: Path, repository: str
) -> dict:
    """Each layout's wheel for the head and for the base. Every layout shares
    one target directory between its two builds, so the base's build reuses the
    head's compiled dependencies; the default layout's is the checkout's own
    ``target/``, which the job's cache keeps, and the others are removed once
    their pair is built, for the runner's disk."""
    result: dict = {"head": head, "base": base, "wheels": {r: {} for r in ROLES}, "reused": {}}
    # Absolute: `git -C repo worktree add` reads a relative path from inside
    # `repo`, so `branch/.vm-checks-base` would land at `branch/branch/...`.
    repo = repo.resolve()
    worktree = repo / ".vm-checks-base"
    try:
        for layout in layouts:
            scratch = layout != "default"
            target = (
                Path(tempfile.mkdtemp(prefix=f"target-{layout}-")) if scratch else repo / "target"
            )
            try:
                name = wheel_name(head, layout)
                result["wheels"]["branch"][layout] = str(
                    build_wheel(repo, layout, target, out / name)
                )
                if base == head:
                    result["wheels"]["base"][layout] = result["wheels"]["branch"][layout]
                    continue
                name = wheel_name(base, layout)
                run = earlier_run(name, repository)
                if run is not None:
                    path = fetch(name, run, repository, fetched / name)
                    result["reused"][name] = run
                else:
                    if not worktree.exists():
                        git(repo, "worktree", "add", "--detach", str(worktree), base)
                    path = build_wheel(worktree, layout, target, out / name)
                result["wheels"]["base"][layout] = str(path)
            finally:
                if scratch:
                    shutil.rmtree(target, ignore_errors=True)
    finally:
        if worktree.exists():
            git(repo, "worktree", "remove", "--force", str(worktree))
    return result


# --- the output digests --------------------------------------------------------------------


def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def make_env(venv: Path, wheel: Path) -> Path:
    """A fresh environment holding ``wheel``, its dependencies and NumPy."""
    _run(["uv", "venv", "--quiet", "--python", "3.12", str(venv)])
    python = _venv_python(venv)
    _run(["uv", "pip", "install", "--quiet", "--python", str(python), str(wheel), "numpy"])
    return python


def moved_columns(case: dict) -> list[str]:
    """The columns of a compared case whose values differ in any row."""
    return sorted(c for c, s in case.get("columns", {}).items() if s.get("rows_differing"))


#: One unit in the last place is between 2^-53 and 2^-52 relative.
ULP = 2.0**-52


def digest_table(report: dict, head: str, base: str) -> str:
    """The Markdown a job summary shows for a ``compare_outputs`` report: per
    case, the columns that moved, the largest relative movement and its size
    in units in the last place (the relative movement over 2^-52, an
    estimate: the tool reports relative movement, not ulps)."""
    lines = [
        "### Output digests, branch against base",
        "",
        f"Branch `{short(head)}` against base `{short(base)}`, both default wheels, "
        "`scripts/compare_outputs.py` (floor "
        f"{report.get('floor', 0):g}). Information only: nothing here fails the run.",
        "",
        "| case | status | columns moved | max relative | ~ulps | rows differ | flips | state |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        if case["status"] != "ok":
            lines.append(
                f"| {case['name']} | {case['status']}: {case.get('why', '')} | | | | | | |"
            )
            continue
        s = case["summary"]
        moved = moved_columns(case)
        shown = ", ".join(f"`{c}`" for c in moved[:8]) + (
            f", +{len(moved) - 8}" if len(moved) > 8 else ""
        )
        flips = sum(s["flipped_columns"].values()) + sum(s["decision_flips"].values())
        lines.append(
            f"| {case['name']} | ok | {shown or 'none'} | {s['max_rel']:.3g} "
            f"| {s['max_rel'] / ULP:.3g} | {s['rows_any_differ']} | {flips} "
            f"| {case.get('state', '-')} |"
        )
    return "\n".join(lines) + "\n"


def run_digests(head_wheel: Path, base_wheel: Path, out: Path, cases: list[str]) -> Path:
    work = Path(tempfile.mkdtemp(prefix="digests-"))
    base_python = make_env(work / "base", base_wheel)
    head_python = make_env(work / "branch", head_wheel)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(head_python), str(REPO / "scripts" / "compare_outputs.py"),
        "--a", str(base_python), "--b", str(head_python),
        "--a-label", "base", "--b-label", "branch",
        "--builtin", "--time-rounds", "0",
        "--out", str(out / "compare"), "--work", str(work / "work"),
    ]  # fmt: skip
    for c in cases:
        cmd += ["--cases", c]
    subprocess.run(cmd, check=True)  # an error fails the run; a movement does not
    shutil.rmtree(work, ignore_errors=True)
    return out / "compare" / "report.json"


def write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


# --- the tests, natively and under qemu ------------------------------------------------------


RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")


def parse_counts(output: str) -> dict[str, int]:
    """The passed, failed and ignored counts, summed over every test binary's
    ``test result`` line in ``output``."""
    found = [tuple(int(n) for n in m) for m in RESULT.findall(output)]
    return {k: sum(f[i] for f in found) for i, k in enumerate(("passed", "failed", "ignored"))}


def built_test_binaries(cargo_json: str) -> list[tuple[Path, Path]]:
    """Each test binary a ``cargo test --no-run --message-format=json`` built,
    with the package directory cargo runs it from."""
    out = []
    for line in cargo_json.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        is_test = msg.get("profile", {}).get("test")
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") and is_test:
            out.append((Path(msg["executable"]), Path(msg["manifest_path"]).parent))
    return out


def verdict(row: dict) -> str | None:
    """Why a run fails, or None: a nonzero exit, a failed test, or no test run
    at all (``cargo test`` and a bare test binary exit 0 on a filter that
    matches nothing)."""
    if row["exit"] != 0:
        return f"exited {row['exit']}"
    if row["failed"]:
        return f"{row['failed']} failed"
    if row["passed"] == 0:
        return "the filter matched no test"
    return None


def _stream(cmd: list[str], **kw) -> tuple[int, str]:
    """Run ``cmd``, echoing its output as it comes, and return it."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", **kw
    )
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return proc.wait(), "".join(lines)


def run_tests(repo: Path, package: str, filters: list[str], qemu_cpu: str = "") -> list[dict]:
    """The package's tests matching ``filters`` in release mode, natively, and
    under qemu posing as ``qemu_cpu`` when given."""
    rows = []
    base = ["cargo", "test", "-p", package, "--release", "--locked"]
    code, text = _stream([*base, *filters, "--", "--include-ignored"], cwd=repo)
    rows.append({"run": "native", "exit": code, **parse_counts(text)})
    if qemu_cpu:
        qemu = shutil.which("qemu-x86_64")
        if qemu is None:
            raise RuntimeError("qemu-x86_64 is not installed")
        built = subprocess.run(
            [*base, "--no-run", "--message-format=json"],
            cwd=repo, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )  # fmt: skip
        total = {"passed": 0, "failed": 0, "ignored": 0}
        worst = 0
        for exe, cwd in built_test_binaries(built.stdout):
            code, text = _stream(
                [qemu, "-cpu", qemu_cpu, str(exe), *filters, "--include-ignored"], cwd=cwd
            )
            worst = worst or code
            for k, v in parse_counts(text).items():
                total[k] += v
        rows.append({"run": f"qemu -cpu {qemu_cpu}", "exit": worst, **total})
    return rows


def tests_table(package: str, filters: list[str], rows: list[dict]) -> str:
    lines = [
        "### Tests",
        "",
        f"`cargo test -p {package} --release {shlex.join(filters)} -- --include-ignored`, "
        "the same test binaries each time.",
        "",
        "| run | passed | failed | ignored | result |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['passed']} | {r['failed']} | {r['ignored']} "
            f"| {verdict(r) or 'ok'} |"
        )
    return "\n".join(lines) + "\n"


# --- the command line --------------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    layouts = parse_layouts(args.variants, args.digests)
    head, base = plan(args.repo, args.base)
    print(f"branch {head}\nbase   {base}\nvariants {','.join(layouts)}")
    write_output({"head": head, "base": base, "head8": short(head), "base8": short(base)})
    return 0


def _cmd_wheels(args: argparse.Namespace) -> int:
    layouts = parse_layouts(args.variants, args.digests)
    head, base = plan(args.repo, args.base)
    repository = os.environ.get("GITHUB_REPOSITORY") or args.repository
    result = make_wheels(args.repo, layouts, head, base, args.out, args.fetched, repository)
    args.manifest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


def _cmd_digests(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wheels = manifest["wheels"]
    report = run_digests(
        Path(wheels["branch"]["default"]),
        Path(wheels["base"]["default"]),
        args.out,
        args.cases,
    )
    table = digest_table(
        json.loads(report.read_text(encoding="utf-8")), manifest["head"], manifest["base"]
    )
    (args.out / "digests.md").write_text(table, encoding="utf-8")
    print(table)
    write_summary(table)
    return 0


def _cmd_tests(args: argparse.Namespace) -> int:
    filters = shlex.split(args.filter)
    if not filters:
        raise SystemExit("tests: an empty filter would run every test; name one")
    rows = run_tests(args.repo, args.package, filters, args.qemu_cpu)
    table = tests_table(args.package, filters, rows)
    print(table)
    write_summary(table)
    bad = [(r["run"], verdict(r)) for r in rows if verdict(r)]
    for run, why in bad:
        print(f"::error::{run}: {why}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--repo", type=Path, default=REPO, help="the branch's checkout, not this script's"
        )
        p.add_argument(
            "--base", default="", help="the base: default the merge-base with origin/main"
        )
        p.add_argument("--variants", default="default,aligned")
        p.add_argument("--digests", action="store_true", help="digests follow: default is needed")

    p = sub.add_parser("plan", help="the branch and base shas, to $GITHUB_OUTPUT")
    common(p)
    p.set_defaults(fn=_cmd_plan)
    w = sub.add_parser("wheels", help="build or fetch each requested wheel, branch and base")
    common(w)
    w.add_argument("--out", type=Path, default=Path("dist"), help="where built wheels go")
    w.add_argument("--fetched", type=Path, default=Path("fetched"), help="where reused ones go")
    w.add_argument("--manifest", type=Path, default=Path("wheels.json"))
    w.add_argument("--repository", default="hgilde/polars-online")
    w.set_defaults(fn=_cmd_wheels)
    d = sub.add_parser("digests", help="compare_outputs on the two default wheels")
    d.add_argument("--manifest", type=Path, default=Path("wheels.json"))
    d.add_argument("--out", type=Path, default=Path("digests"))
    d.add_argument(
        "--cases", action="append", default=[], help="a case file or folder, as compare_outputs"
    )
    d.set_defaults(fn=_cmd_digests)
    t = sub.add_parser("tests", help="a package's tests by filter, natively and under qemu")
    t.add_argument("--repo", type=Path, default=REPO, help="the branch's checkout")
    t.add_argument("--package", default="online-core")
    t.add_argument("--filter", required=True, help="cargo test's filter (words: several filters)")
    t.add_argument("--qemu-cpu", default="", help="also run under qemu-x86_64 -cpu this CPU")
    t.set_defaults(fn=_cmd_tests)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
