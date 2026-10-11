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
/// C-2: a relative path in the TOML file is read against the file's own
/// directory, so a config runs the same from anywhere; a path given as a
/// flag is read against the working directory, as a shell user expects.
#[test]
fn a_relative_path_is_the_config_files_in_the_toml_and_the_working_directorys_in_a_flag() {
    let dir = fresh_dir("relative");
    let (conf, work) = (dir.join("conf"), dir.join("work"));
    std::fs::create_dir_all(&conf).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    write_input(&conf.join("in.parquet"), 40);
    let cfg = job(
        &conf,
        "in.parquet",
        "out.parquet",
        "save_state = 'run.state'",
    );

    let (code, _, err) = run(cli()
        .current_dir(&work)
        .args(["--config", text(&cfg), "-q"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(conf.join("out.parquet").is_file() && conf.join("run.state").is_file());
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0);

    std::fs::copy(conf.join("in.parquet"), work.join("flag-in.parquet")).unwrap();
    let (code, _, err) = run(cli().current_dir(&work).args([
        "--config",
        text(&cfg),
        "-q",
        "--input",
        "flag-in.parquet",
        "--output",
        "flag-out.parquet",
        "--save-state",
        "flag.state",
    ]));
    assert_eq!(code, Some(0), "{err}");
    assert!(work.join("flag-out.parquet").is_file() && work.join("flag.state").is_file());
    assert!(!conf.join("flag-out.parquet").exists());

    // A config named by a relative path is found from the working
    // directory, and its own paths still from where it sits.
    let (code, _, err) =
        run(cli()
            .current_dir(&dir)
            .args(["--config", "conf/job.toml", "-q", "--no-output"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(!dir.join("run.state").exists());
    std::fs::remove_dir_all(&dir).unwrap();
}
/// C-9: the progress line is drawn only when standard error is a terminal,
/// so a log or a pipe carries none of its carriage returns; and the summary
/// counts in the singular at one.
#[test]
fn the_progress_is_for_a_terminal_and_the_summary_counts_in_the_singular() {
    let dir = fresh_dir("progress");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    for (rows, chunk_size, said) in [
        (50, 100, "50 rows (1 chunk)"),
        (50, 7, "50 rows (8 chunks)"),
        (1, 100, "1 row (1 chunk)"),
    ] {
        write_input(&input, rows);
        let cfg = job(
            &dir,
            text(&input),
            text(&output),
            &format!("chunk_size = {chunk_size}"),
        );
        let (code, stdout, err) = run(cli().args(["--config", text(&cfg)]));
        assert_eq!(code, Some(0), "{err}");
        assert!(!err.contains("rows in") && !err.contains('\r'), "{err:?}");
        let wrote = format!("wrote {said} to {}", output.display());
        assert!(
            stdout.lines().any(|l| l == wrote),
            "{wrote:?} in {stdout:?}"
        );
        let (code, stdout, err) = run(cli().args([
            "--config",
            text(&cfg),
            "--no-output",
            "--save-state",
            text(&dir.join("s.state")),
        ]));
        assert_eq!(code, Some(0), "{err}");
        let processed = format!("processed {said}, no output");
        assert!(
            stdout.lines().any(|l| l == processed),
            "{processed:?} in {stdout:?}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// C-11: `--version` names what a state file and a bug report need beside
/// the package: the state schema this build writes, and the Rust Polars it
/// carries. `-V` stays the one line.
#[test]
fn version_names_the_state_schema_and_the_rust_polars() {
    let (code, stdout, err) = run(cli().arg("--version"));
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(
        stdout,
        format!(
            "polars-online {}\nstate schema {}\nRust Polars {}\n",
            env!("CARGO_PKG_VERSION"),
            online_polars::online_core::SCHEMA_VERSION,
            polars::VERSION,
        )
    );
    let (code, stdout, err) = run(cli().arg("-V"));
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(
        stdout,
        format!("polars-online {}\n", env!("CARGO_PKG_VERSION"))
    );
}

/// A `chunk_size` of 0 is refused in the words Python uses for it, from the
/// TOML file and from the flag alike.
#[test]
fn a_chunk_size_of_0_is_refused_as_python_refuses_it() {
    let dir = fresh_dir("chunk0");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 10);
    for (extra, flag) in [
        ("chunk_size = 0", &[][..]),
        ("", &["--chunk-size", "0"][..]),
    ] {
        let cfg = job(&dir, text(&input), text(&output), extra);
        let (code, _, err) = run(cli().args(["--config", text(&cfg)]).args(flag));
        assert_eq!(code, Some(1), "{err}");
        assert!(
            err.contains("chunk_size must be at least 1, got 0"),
            "{extra} {flag:?}: {err}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
