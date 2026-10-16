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

/// `--threads` and the TOML's `threads` (task 253): the flag overrides the
/// key, as every flag does; 0 is refused in the words Python uses, from
/// either; a run on a pool of its own writes the frame the default pool
/// writes; and given a count, the run never reads
/// `POLARS_ONLINE_MAX_THREADS`, which still sizes, and so refuses, the
/// default pool.
#[test]
fn threads_from_the_flag_or_the_key_and_the_variable_only_without_them() {
    let dir = fresh_dir("threads");
    let input = dir.join("in.parquet");
    write_input(&input, 300);
    for (extra, flag) in [("threads = 0", &[][..]), ("", &["--threads", "0"][..])] {
        let cfg = job(&dir, text(&input), text(&dir.join("o.parquet")), extra);
        let (code, _, err) = run(cli().args(["--config", text(&cfg)]).args(flag));
        assert_eq!(code, Some(1), "{err}");
        assert!(
            err.contains("threads=0 is not a number of threads"),
            "{extra} {flag:?}: {err}"
        );
    }
    let cfg = job(
        &dir,
        text(&input),
        text(&dir.join("o.parquet")),
        "threads = 3",
    );
    let (code, out, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("threads: 3"), "{out}");
    let (code, out, err) = run(cli().args(["--config", text(&cfg), "--dry-run", "--threads", "2"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(
        out.contains("threads: 2") && !out.contains("threads: 3"),
        "{out}"
    );

    let read = |p: &Path| {
        ParquetReader::new(std::fs::File::open(p).unwrap())
            .finish()
            .unwrap()
    };
    let default_out = dir.join("default.parquet");
    let cfg = job(&dir, text(&input), text(&default_out), "");
    let (code, _, err) = run(cli().args(["--config", text(&cfg)]));
    assert_eq!(code, Some(0), "{err}");
    for n in ["1", "2", "8"] {
        let out = dir.join(format!("t{n}.parquet"));
        let cfg = job(&dir, text(&input), text(&out), "");
        let (code, _, err) = run(cli()
            .args(["--config", text(&cfg), "--threads", n])
            .env("POLARS_ONLINE_MAX_THREADS", "eight"));
        assert_eq!(code, Some(0), "--threads {n}: {err}");
        assert!(
            read(&out).equals_missing(&read(&default_out)),
            "--threads {n}"
        );
    }
    let (code, _, err) = run(cli()
        .args(["--config", text(&cfg)])
        .env("POLARS_ONLINE_MAX_THREADS", "eight"));
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("POLARS_ONLINE_MAX_THREADS=\"eight\""), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---- Task 252: command-line paths no test reached. ----

/// A config that does not parse and holds a backslash is told how a
/// Windows path is written in TOML; one whose specs give a model that is
/// not a table, or a table with no `type`, is refused with TOML's message
/// and no key it cannot name.
#[test]
fn a_config_that_does_not_parse_is_told_what_it_can_be() {
    let dir = fresh_dir("parse-hints");
    let cfg = dir.join("job.toml");
    std::fs::write(
        &cfg,
        "input = \"C:\\Users\\in.parquet\"\noutput = \"out.parquet\"\n",
    )
    .unwrap();
    let (code, _, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("hint: a backslash starts an escape sequence in a TOML basic string")
            && err.contains("input = 'C:\\data\\in.parquet'"),
        "{err}"
    );
    std::fs::write(
        &cfg,
        "input = 'in.parquet'\noutput = 'out.parquet'\n[[specs]]\nname = \"a\"\n\
         targets = [\"y\"]\nfeatures = [\"x0\"]\nhalf_life = 20.0\nmodel = \"ewridge\"\n\
         [[specs]]\nname = \"b\"\ntargets = [\"y\"]\nfeatures = [\"x0\"]\nhalf_life = 20.0\n\
         [specs.model]\nridge = 1.0\n",
    )
    .unwrap();
    let (code, _, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("parsing ") && !err.contains("\nat spec"),
        "{err}"
    );
    assert!(!err.contains("hint: a backslash"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A model key deprecated in one model's table is read as its new name,
/// with a warning that says so, and the run goes on.
#[test]
fn a_deprecated_model_key_is_read_with_a_warning() {
    let dir = fresh_dir("deprecated-model-key");
    let input = dir.join("in.parquet");
    write_input(&input, 60);
    let cfg = dir.join("job.toml");
    std::fs::write(
        &cfg,
        format!(
            "input = '{}'\noutput = '{}'\n[[specs]]\nname = \"cp\"\nfeatures = [\"x0\"]\n\
             [specs.model]\ntype = \"bocpd\"\nemission = \"gaussian\"\n",
            text(&input),
            text(&dir.join("out.parquet"))
        ),
    )
    .unwrap();
    let (code, out, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(
        err.contains(
            "polars-online: warning: emission = \"gaussian\" is deprecated: it is \
             covariance_type = \"full\""
        ),
        "{err}"
    );
    assert!(out.starts_with("config OK: 1 spec(s)"), "{out}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `kmeans` reading `x0`, its clusters under the deprecated `k` or as
/// `n_clusters`, with `half_life = <half_life>` written as given: a config
/// for `input`, writing `output`.
fn kmeans_job(dir: &Path, input: &Path, output: &Path, clusters: &str, half_life: &str) -> PathBuf {
    let cfg = dir.join("job.toml");
    std::fs::write(
        &cfg,
        format!(
            "input = '{}'\noutput = '{}'\n[[specs]]\nname = \"km\"\nfeatures = [\"x0\"]\n\
             half_life = {half_life}\n[specs.model]\ntype = \"kmeans\"\n{clusters} = 2\n",
            text(input),
            text(output)
        ),
    )
    .unwrap();
    cfg
}

/// Task 269, D-2: a deprecated model key was forwarded by a round trip
/// through JSON, which has no infinity, so a TOML float `inf` anywhere in
/// the file became null and the config was refused ("invalid type: null").
/// Forwarded over TOML's own values, the run reads it as the infinity it
/// is: the output is the one the new name writes.
#[test]
fn a_toml_infinity_beside_a_deprecated_model_key_is_kept() {
    let dir = fresh_dir("deprecated-inf");
    let input = dir.join("in.parquet");
    write_input(&input, 60);
    let (old, new) = (dir.join("old.parquet"), dir.join("new.parquet"));
    let cfg = kmeans_job(&dir, &input, &old, "k", "inf");
    let (code, out, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(
        err.contains("warning: k is deprecated: it was renamed n_clusters"),
        "{err}"
    );
    assert!(out.starts_with("config OK: 1 spec(s)"), "{out}");
    let (code, _, err) = run(cli().args(["--config", text(&cfg)]));
    assert_eq!(code, Some(0), "{err}");
    let cfg = kmeans_job(&dir, &input, &new, "n_clusters", "inf");
    let (code, _, err) = run(cli().args(["--config", text(&cfg)]));
    assert_eq!(code, Some(0), "{err}");
    assert!(!err.contains("deprecated"), "{err}");
    // And a finite half-life writes another frame, so the two agreeing is
    // the infinity read, not a half-life ignored.
    let finite = dir.join("finite.parquet");
    let cfg = kmeans_job(&dir, &input, &finite, "n_clusters", "5.0");
    let (code, _, err) = run(cli().args(["--config", text(&cfg)]));
    assert_eq!(code, Some(0), "{err}");
    let read = |p: &Path| {
        ParquetReader::new(std::fs::File::open(p).unwrap())
            .finish()
            .unwrap()
    };
    assert!(read(&old).equals_missing(&read(&new)));
    assert!(!read(&old).equals_missing(&read(&finite)));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 269, D-7: a config with a deprecated key was rewritten, and a parse
/// error after it quoted the rewritten text's line; it names the file's own
/// line and quotes it, as it does with nothing to forward.
#[test]
fn a_parse_error_beside_a_deprecated_key_names_the_files_own_line() {
    let dir = fresh_dir("deprecated-line");
    let input = dir.join("in.parquet");
    let output = dir.join("out.parquet");
    for clusters in ["n_clusters", "k"] {
        let cfg = kmeans_job(&dir, &input, &output, clusters, "\"ten\"");
        // Two lines of comment above it, which a rewrite drops.
        let body = std::fs::read_to_string(&cfg).unwrap();
        std::fs::write(&cfg, format!("# a job\n# of one spec\n{body}")).unwrap();
        let (code, _, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
        assert_eq!(code, Some(1), "{err}");
        // The file's eighth line, `half_life = "ten"`.
        assert!(
            err.contains("line 8") && err.contains("half_life = \"ten\""),
            "{clusters}: {err}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 269, D-5: a `threads` that is no positive integer -- negative, a
/// fraction, a word, a bool -- is refused in the words 0 is, and Python's
/// surfaces use, not by serde's "expected usize".
#[test]
fn a_threads_that_is_no_count_is_refused_in_the_one_message() {
    let dir = fresh_dir("threads-type");
    let input = dir.join("in.parquet");
    write_input(&input, 10);
    for (value, shown) in [
        ("-1", "-1"),
        ("2.5", "2.5"),
        ("\"auto\"", "\"auto\""),
        ("true", "true"),
    ] {
        let extra = format!("threads = {value}");
        let cfg = job(&dir, text(&input), text(&dir.join("o.parquet")), &extra);
        let (code, _, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
        assert_eq!(code, Some(1), "{err}");
        assert!(
            err.contains(&format!(
                "threads={shown} is not a number of threads: give an integer of 1 or more"
            )),
            "{value}: {err}"
        );
        assert!(!err.contains("expected usize"), "{value}: {err}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A dry run of a run with no output names what the run's products are,
/// the state and the closed groups, and where the closed groups go.
#[test]
fn a_dry_run_with_no_output_names_the_runs_products() {
    let dir = fresh_dir("dry-no-output");
    let input = dir.join("in.parquet");
    let mut df = df!(
        "g" => (0..20i64).map(|i| i / 5).collect::<Vec<_>>(),
        "x0" => (0..20).map(|i| f64::from(i % 5)).collect::<Vec<_>>(),
        "y" => (0..20).map(|i| f64::from(i % 3)).collect::<Vec<_>>()
    )
    .unwrap();
    ParquetWriter::new(std::fs::File::create(&input).unwrap())
        .finish(&mut df)
        .unwrap();
    let (state, closed) = (dir.join("state.bin"), dir.join("closed.parquet"));
    let cfg = dir.join("job.toml");
    std::fs::write(
        &cfg,
        format!(
            "input = '{}'\nsave_state = '{}'\nclosed_groups = '{}'\n[[specs]]\nname = \"c\"\n\
             features = [\"x0\", \"y\"]\ngroup = \"g\"\nhalf_life = 10.0\n\
             group_close = \"monotone\"\n[specs.model]\ntype = \"ew_cov\"\n",
            text(&input),
            text(&state),
            text(&closed)
        ),
    )
    .unwrap();
    let (code, out, err) = run(cli().args(["--config", text(&cfg), "--dry-run"]));
    assert_eq!(code, Some(0), "{err}");
    assert!(
        out.contains(&format!(
            "output: none (--no-output); the run's product is the state, {}, and the closed \
             groups, {}",
            state.display(),
            closed.display()
        )),
        "{out}"
    );
    assert!(
        out.contains(&format!("closed groups: {} (parquet)", closed.display())),
        "{out}"
    );
    assert!(
        !state.exists() && !closed.exists(),
        "a dry run writes nothing"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
