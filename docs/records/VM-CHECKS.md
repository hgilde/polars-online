<!-- kind: record -->
# VM checks: the wheels and the x86 checks a runner builds for the VMs

Task 466 (the user, 2026-11-18). The x86-64 VMs time wall-clock. Building the
release wheels they time took each of them 15 to 20 minutes a build, and a
restart (they restart often) lost the build. `.github/workflows/vm-checks.yml`
builds them on a GitHub runner instead, and the VM downloads them. The script
behind it is `scripts/vm_checks.py`; `tests/test_vm_checks.py` holds both.

The scope is what a VM would otherwise do itself, and no more. Four things
were considered and left out, each at the user's word the same day:

- a trigger on a push to `vm/**`: a push is no request, and a build nobody
  asked for saves nobody's time, so the workflow is dispatched by hand, by
  the coordinator when it opens a VM request;
- a matrix of jobs: a request holds one runner, one step after another,
  because the cap on concurrent jobs is shared with CI;
- a Callgrind comparison of the branch with its base: `budgets.yml` counts
  instructions on every push to main, so a pre-merge count repeats it;
- `scripts/release_probe.py`: it is the workload `compare_release.py` runs
  against a *released* version from PyPI, not a comparison of two builds, so
  it has nothing to say about a branch and its base.

## Dispatching

From any checkout with `gh`; the workflow file is on `main`, so dispatch on
`main` and name the branch in `ref` (a vm branch cut before the workflow
existed has no copy of it):

```
gh workflow run vm-checks.yml --ref main -f ref=vm/x -f variants=default,aligned
gh workflow run vm-checks.yml --ref main -f ref=vm/x -f variants=default,aligned,v3 \
    -f base=<sha-or-branch> -f digests=true
gh workflow run vm-checks.yml --ref main -f ref=vm/x -f tests=solve::small_tests \
    -f test_package=online-core -f qemu_cpu=Nehalem
```

| input | default | meaning |
|---|---|---|
| `ref` | (required) | the branch, tag or full sha to build |
| `variants` | `default,aligned` | any of `default`, `aligned`, `v3`; a VM request always times both layouts, so those two are the default |
| `base` | the merge-base of `ref` with `origin/main` | a sha or a branch, to build the base from |
| `digests` | `false` | also compare the two default wheels' outputs |
| `tests` | empty (skip) | a cargo test filter to run on x86-64 |
| `test_package` | `online-core` | the package the filter's tests are in |
| `qemu_cpu` | empty (skip) | also run those tests under `qemu-x86_64 -cpu` this CPU, e.g. `Nehalem`, which has no AVX2 or FMA |

The three layouts, all release wheels of the same source:

- `default`: as the release wheel is built;
- `aligned`: `RUSTFLAGS="-C llvm-args=-align-all-functions=6"`, every
  function on a 64-byte boundary, a second code layout (a layout alone moves
  untouched cases, so a slowdown counts only if it persists across both);
- `v3`: `RUSTFLAGS="-C target-cpu=x86-64-v3"`, the instruction set of PLAN
  task 448's measurement.

Every build also names lld in `RUSTFLAGS` (the build-prep action's
`.cargo/config.toml` is replaced by that variable; the tests, which set no
`RUSTFLAGS`, find it by cargo's walk up from `branch/` to the workspace root), so the only
difference between layouts is the flag under test.

A second dispatch for the same `ref` cancels the first (`cancel-in-progress`,
group `vm-checks-<ref>`); dispatches for different branches run side by side.

## What a run does

One job, `checks`, 120 minutes at most, on `ubuntu-24.04` (the image
`budgets.yml` names). In order:

1. checks out the commit the workflow runs from (`github.sha`, main when
   dispatched there) at the workspace root, and `ref` with its whole history
   in `branch/`. The tools (`scripts/vm_checks.py`, `compare_outputs.py`, the
   build-prep action) come from the first, because a `vm/` branch is cut from
   an older main and may predate them (run 44309746665 died on that: the
   script was not in the branch); the second is what is built, merge-based
   and tested, each `vm_checks.py` command taking `--repo branch`. Then it
   frees the runner's disk and installs lld (`.github/actions/linux-build-prep`, plus `qemu-user` when
   `qemu_cpu` is set), the toolchain, a cargo cache (of `branch/`'s workspace)
   and `uv`;
2. `scripts/vm_checks.py plan`: the branch's head sha and the base's, short
   (8 hex digits) for the artifact names; an unknown variant or base fails
   here, before any building;
3. `scripts/vm_checks.py wheels`: for each layout, in the order given, the
   branch's wheel and then the base's. The base's wheel is **reused** when an
   earlier run uploaded an unexpired artifact called `wheel-<sha8>-<layout>`
   (looked up through the REST API, newest first) and built otherwise, in a
   `git worktree` of the base, sharing the layout's target directory so the
   second build reuses the first's compiled dependencies. A branch at its
   base builds once. The wheels are built natively, `maturin build --release
   --locked --compatibility linux`: the file is tagged `linux_x86_64`, not
   manylinux, so **it needs the runner's glibc (2.39) or newer**, and
   `uv pip install` takes it on any x86-64 Linux with that;
4. uploads each wheel built in this run (a reused base wheel is not
   uploaded again; it is already an artifact);
5. with `digests: true`: `scripts/vm_checks.py digests`, which installs the
   two default wheels into fresh environments (with NumPy and whatever Polars
   resolves, the same for both) and runs `scripts/compare_outputs.py` over
   its cases (the built-in smoke cases; its `--cases` takes more files),
   then uploads the report. It is information: a column that moved is in the
   table and the artifact, and fails nothing; an error in the run does;
6. with `tests` set: `scripts/vm_checks.py tests`, which runs
   `cargo test -p <test_package> --release <tests> -- --include-ignored`
   natively on the runner, and with `qemu_cpu` set, builds the same test
   binaries once (`cargo test --no-run --message-format=json`, taking each
   binary's path from the output) and runs each under
   `qemu-x86_64 -cpu <qemu_cpu>` with the same filter. The build has the
   debug assertions and overflow checks on, as `cpu-paths.yml`'s, so a test
   that expects one to fire passes. Passed, failed and ignored counts go in
   the job summary, a failure fails the run (these are correctness tests),
   and so does a filter that matches no test (`cargo test` exits 0 on one;
   the count is checked). Doctests run natively only;
7. keeps the manifest.

## Artifacts

| name | holds |
|---|---|
| `wheel-<sha8>-default` | one `polars_online-*.whl`, the layout's wheel of that commit |
| `wheel-<sha8>-aligned` | the same, aligned |
| `wheel-<sha8>-v3` | the same, x86-64-v3 |
| `manifest-<head sha8>` | `wheels.json`: both full shas, every wheel's path in the run, and `reused`: each base wheel taken from an earlier run, with that run's id |
| `digests-<head sha8>-vs-<base sha8>` | `compare/report.json` (every case, every column's statistics), `compare/summary.md`, `digests.md` (the job summary's table) |

`<sha8>` is the first eight characters of the commit's sha, the branch's head
for the branch's wheels and the base for the base's. A VM times branch
against base without building anything: it needs `wheel-<head8>-<layout>`
and `wheel-<base8>-<layout>` for each layout, wherever the base's was built.
The digest table's "~ulps" is the largest relative movement over 2^-52, an
estimate: `compare_outputs.py` reports relative movement, not units in the
last place.

Artifacts keep 90 days (GitHub's default), and a base wheel older than that
is rebuilt by the next run that needs it.

## Fetching, as a VM does

The head sha of the branch is `git rev-parse origin/vm/x`; the base's is the
merge-base with `origin/main`, or what `base` was given. The run's title is
`VM checks <ref> (<variants>)`.

With `gh` (authenticated):

```
gh run list --workflow vm-checks.yml --limit 20 \
    --json databaseId,displayTitle,status,conclusion \
    --jq '.[] | select(.displayTitle | startswith("VM checks vm/x "))'
gh run watch <run id> --exit-status           # if it is still running
gh run download <run id> --name wheel-<head8>-default --dir wheels/branch-default
gh run download <run id> --name wheel-<head8>-aligned --dir wheels/branch-aligned
# the base's wheel may belong to an earlier run: read the manifest's `reused`,
# or ask for the artifact by name, whichever run holds it
gh api "repos/hgilde/polars-online/actions/artifacts?name=wheel-<base8>-default" \
    --jq '.artifacts[] | select(.expired | not) | [.workflow_run.id, .created_at] | @tsv'
gh run download <that run id> --name wheel-<base8>-default --dir wheels/base-default
```

Looking an artifact up by its name needs no run id and is the robust form:
the name is a function of the sha and the layout alone.

Without `gh`, with a token (a fine-grained one with read access to Actions on
this repository, or a classic one with `repo`; artifact downloads need a
token even on a public repository):

```
API=https://api.github.com/repos/hgilde/polars-online
H1="Authorization: Bearer $GITHUB_TOKEN"; H2="Accept: application/vnd.github+json"
curl -fsS -H "$H1" -H "$H2" "$API/actions/artifacts?name=wheel-<sha8>-default&per_page=100" \
    | python3 -c 'import json,sys; a=[x for x in json.load(sys.stdin)["artifacts"] if not x["expired"]]; print(max(a, key=lambda x: x["created_at"])["archive_download_url"])'
curl -fsSL -H "$H1" -H "$H2" -o wheel.zip "<the archive_download_url above>"
unzip -o wheel.zip -d wheels/branch-default
```

(`archive_download_url` redirects to a signed storage URL; `-L` follows it,
and curl drops the `Authorization` header on the redirect, as it must.) The
runs themselves: `GET $API/actions/workflows/vm-checks.yml/runs`, each with
its `display_title`, `status` and `conclusion`.

Then, in a fresh environment per wheel, `uv pip install wheels/branch-default/*.whl`
and the same for the base, and time as the request says.

## What could not be verified without a push

The workflow has been linted (`actionlint`) and parsed, and the script's logic
tested with the commands it runs faked; it has not run on a GitHub runner.
The first dispatch is the test of: `maturin build --compatibility linux`
finding the interpreter (`uv python find 3.12`) on the image; the artifact
listing's JSON as `earlier_run` reads it; `gh run download` in the job; the
cache restored for the default layout; `qemu-user`'s package name on the image
and `qemu-x86_64 -cpu Nehalem` running the test binaries; and the disk and the
time of six cold builds (`variants=default,aligned,v3` with a base nobody
built).
