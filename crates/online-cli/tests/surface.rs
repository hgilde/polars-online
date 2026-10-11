//! The command line's own surface, task 244: the variables and flags a
//! person types, where a path in the TOML file points, and what the binary
//! prints back. `run.rs` holds the runs themselves.

use std::path::{Path, PathBuf};
use std::process::Command;

use polars::prelude::*;

/// A directory of its own for one test, emptied first.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "polars-online-cli-surface-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `n` rows of a clock, a feature and a target, as parquet.
fn write_input(path: &Path, n: u32) {
    let mut df = df!(
        "t" => (0..n).map(f64::from).collect::<Vec<_>>(),
        "x0" => (0..n).map(|i| f64::from(i % 5) - 2.0).collect::<Vec<_>>(),
        "y" => (0..n).map(|i| f64::from(i % 3)).collect::<Vec<_>>()
    )
    .unwrap();
    ParquetWriter::new(std::fs::File::create(path).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// `job.toml` in `dir`, reading `input` and writing `output` as written
/// (TOML literal strings, so a Windows path needs no escaping), with
/// `extra` top-level keys. Its path.
fn job(dir: &Path, input: &str, output: &str, extra: &str) -> PathBuf {
    let cfg = dir.join("job.toml");
    std::fs::write(
        &cfg,
        format!(
            "input = '{input}'\noutput = '{output}'\n{extra}\n[[specs]]\nname = \"ridge\"\n\
             targets = [\"y\"]\nfeatures = [\"x0\"]\nclock = \"t\"\nhalf_life = 20.0\ngap_cap = 5.0\n\
             [specs.model]\ntype = \"ewridge\"\n"
        ),
    )
    .unwrap();
    cfg
}

/// The binary, with none of the variables these tests set inherited.
fn cli() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_polars-online"));
    for var in [
        "POLARS_ONLINE_TIMING",
        "ONLINE_TIMING",
        "POLARS_ONLINE_WARN_UNSTABLE",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

/// Its exit code, stdout and stderr, both piped: neither is a terminal.
fn run(cmd: &mut Command) -> (Option<i32>, String, String) {
    let out = cmd.output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn text(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// C-1: the timings are on at `POLARS_ONLINE_TIMING=1` and at nothing else,
/// as `POLARS_ONLINE_WARN_UNSTABLE` is. The bank's line per chunk and the
/// runner's line per run both print. The old name, `ONLINE_TIMING`, turns
/// nothing on, and says once what replaced it.
#[test]
fn the_timings_are_on_at_polars_online_timing_1_alone() {
    let dir = fresh_dir("timing");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 50);
    let cfg = job(&dir, text(&input), text(&output), "");
    let stderr = |env: &[(&str, &str)]| {
        let mut cmd = cli();
        cmd.args(["--config", text(&cfg), "-q"])
            .envs(env.iter().copied());
        let (code, _, err) = run(&mut cmd);
        assert_eq!(code, Some(0), "{env:?}: {err}");
        err
    };
    let on = stderr(&[("POLARS_ONLINE_TIMING", "1")]);
    assert!(
        on.lines()
            .any(|l| l.starts_with("POLARS_ONLINE_TIMING rows=50 ")),
        "{on}"
    );
    assert!(
        on.lines()
            .any(|l| l.starts_with("POLARS_ONLINE_TIMING run rows=50 chunks=1 ")),
        "{on}"
    );
    for env in [
        &[][..],
        &[("POLARS_ONLINE_TIMING", "0")][..],
        &[("POLARS_ONLINE_TIMING", "true")][..],
    ] {
        let off = stderr(env);
        assert!(!off.contains("TIMING"), "{env:?}: {off}");
    }
    let old = stderr(&[("ONLINE_TIMING", "1")]);
    let notice = "polars-online: warning: ONLINE_TIMING was renamed POLARS_ONLINE_TIMING; \
                  set POLARS_ONLINE_TIMING=1 for the timings";
    assert_eq!(old.lines().filter(|l| *l == notice).count(), 1, "{old}");
    assert!(
        !old.contains("TIMING rows") && !old.contains("TIMING run"),
        "{old}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
