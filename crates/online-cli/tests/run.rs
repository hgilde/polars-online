//! CLI-level tests: streaming a parquet file through a TOML config, chunk
//! invariance across chunk_size, and resume-from-state (docs/PLAN.md task 15).

use std::path::{Path, PathBuf};

use online_polars::{Bank, RunConfig, run_config};
use polars::prelude::*;

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "polars-online-cli-test-{}-{name}",
        std::process::id()
    ));
    p
}

/// Deterministic two-group stream written to parquet, one row group.
fn write_input(path: &Path, n: usize) -> PolarsResult<()> {
    write_input_in_row_groups(path, n, None)
}

/// The same stream in row groups of `row_group_size` rows.
fn write_input_in_row_groups(
    path: &Path,
    n: usize,
    row_group_size: Option<usize>,
) -> PolarsResult<()> {
    let mut s = 2024u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut group = Vec::new();
    let mut t = Vec::new();
    let (mut x0, mut x1, mut y, mut w) = (vec![], vec![], vec![], vec![]);
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs();
        t.push(clocks[g]);
        let a = lcg();
        let b = lcg();
        x0.push(a);
        x1.push(b);
        y.push(if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg());
        w.push(1.0);
    }
    let mut df = df!(
        "group" => group, "t" => t, "x0" => x0, "x1" => x1, "y" => y, "w" => w
    )?;
    ParquetWriter::new(std::fs::File::create(path)?)
        .with_row_group_size(row_group_size)
        .finish(&mut df)?;
    Ok(())
}

/// A path as a TOML **basic** string: backslashes doubled.
///
/// Without this the test wrote `input = "C:\Users\runner\..."` on Windows,
/// where TOML reads `\U` as the start of a unicode escape and fails with
/// "too few unicode value digits". That is TOML behaving correctly and the
/// caller being wrong -- the same trap any Windows user hand-writing a config
/// falls into, which is why `main.rs` now says so in the error.
fn toml_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn config(input: &Path, output: &Path, chunk_size: usize) -> RunConfig {
    let toml = format!(
        r#"
input = "{}"
output = "{}"
chunk_size = {chunk_size}

[[specs]]
name = "ridge"
targets = ["y"]
features = ["x0", "x1"]
clock = "t"
half_life = 50.0
gap_cap = 10.0
weight = "w"
group = "group"
min_weight = 5.0

[specs.model]
type = "ewridge"
ridge = 1e-6
max_rows_between_solves = 1
"#,
        toml_path(input),
        toml_path(output)
    );
    toml::from_str(&toml).unwrap_or_else(|e| panic!("test wrote invalid TOML: {e}\n{toml}"))
}

fn read_preds(path: &Path) -> PolarsResult<Vec<Option<f64>>> {
    let df = ParquetReader::new(std::fs::File::open(path)?).finish()?;
    let s = df.column("ridge")?.struct_()?.field_by_name("pred_y")?;
    Ok(s.f64()?.iter().collect())
}

#[test]
fn streams_parquet_and_is_chunk_invariant() {
    let input = tmp("in.parquet");
    write_input(&input, 2000).unwrap();

    let out_a = tmp("a.parquet");
    let out_b = tmp("b.parquet");
    let sa = run_config(&config(&input, &out_a, 5000), |_| Ok(())).unwrap();
    let sb = run_config(&config(&input, &out_b, 137), |_| Ok(())).unwrap();

    assert_eq!(sa.rows, 2000);
    assert_eq!(sb.rows, 2000);
    assert_eq!(sa.chunks, 1);
    assert!(sb.chunks > 10);
    // Chunking must not change a single number (docs/PLAN.md §9 class 2).
    assert_eq!(read_preds(&out_a).unwrap(), read_preds(&out_b).unwrap());

    for p in [&input, &out_a, &out_b] {
        let _ = std::fs::remove_file(p);
    }
}

/// A chunk that straddles a row-group boundary is a multi-chunk frame, and
/// the bank's outputs are single-chunk. The batched parquet writer walks the
/// columns' chunks in lockstep and only `debug_assert`s that they line up:
/// here (a debug build) that assertion fired; in release the mismatch was a
/// panic inside arrow's record-batch constructor -- on every file whose row
/// groups were not a multiple of `chunk_size`.
#[test]
fn row_groups_need_not_align_with_chunk_size() {
    let aligned = tmp("rg-one.parquet");
    let split = tmp("rg-50.parquet");
    write_input(&aligned, 1000).unwrap();
    write_input_in_row_groups(&split, 1000, Some(50)).unwrap();

    let out_a = tmp("rg-one-out.parquet");
    let out_b = tmp("rg-50-out.parquet");
    let sa = run_config(&config(&aligned, &out_a, 80), |_| Ok(())).unwrap();
    let sb = run_config(&config(&split, &out_b, 80), |_| Ok(())).unwrap();

    assert_eq!((sa.rows, sa.chunks), (1000, 13));
    assert_eq!((sb.rows, sb.chunks), (1000, 13));
    assert_eq!(read_preds(&out_a).unwrap(), read_preds(&out_b).unwrap());
    let df = ParquetReader::new(std::fs::File::open(&out_b).unwrap())
        .finish()
        .unwrap();
    assert_eq!(df.height(), 1000);
    assert_eq!(df.column("y").unwrap().null_count(), 0);

    for p in [&aligned, &split, &out_a, &out_b] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn resume_from_state_continues_the_stream() {
    let input = tmp("resume-in.parquet");
    write_input(&input, 1000).unwrap();

    // Reference: the whole file in one go.
    let full_out = tmp("resume-full.parquet");
    run_config(&config(&input, &full_out, 100_000), |_| Ok(())).unwrap();
    let full = read_preds(&full_out).unwrap();

    // Split the input in two, run the first half, save, resume on the second.
    let half_a = tmp("resume-a.parquet");
    let half_b = tmp("resume-b.parquet");
    {
        let df = ParquetReader::new(std::fs::File::open(&input).unwrap())
            .finish()
            .unwrap();
        let mut a = df.slice(0, 500);
        let mut b = df.slice(500, 500);
        ParquetWriter::new(std::fs::File::create(&half_a).unwrap())
            .finish(&mut a)
            .unwrap();
        ParquetWriter::new(std::fs::File::create(&half_b).unwrap())
            .finish(&mut b)
            .unwrap();
    }

    let state = tmp("resume.state");
    let out_a = tmp("resume-out-a.parquet");
    let mut cfg_a = config(&half_a, &out_a, 100_000);
    cfg_a.save_state = Some(state.clone());
    run_config(&cfg_a, |_| Ok(())).unwrap();

    let out_b = tmp("resume-out-b.parquet");
    let mut cfg_b = config(&half_b, &out_b, 100_000);
    cfg_b.load_state = Some(state.clone());
    run_config(&cfg_b, |_| Ok(())).unwrap();

    let mut resumed = read_preds(&out_a).unwrap();
    resumed.extend(read_preds(&out_b).unwrap());
    assert_eq!(resumed, full, "resuming must reproduce the unbroken run");

    for p in [&input, &full_out, &half_a, &half_b, &state, &out_a, &out_b] {
        let _ = std::fs::remove_file(p);
    }
}

/// `stats = []` on `ew_cov` from TOML (docs/ENHANCEMENTS.md E43, task 36):
/// the run writes a struct holding `n_eff` alone, and the saved state carries
/// the Gram the run accumulated. The same TOML with `stats` missing still
/// means mean/std/corr -- the empty list is explicit. `targets` mirrors
/// `features[0]`, as `po.spec.ew_cov` fills it in (`ModelKind::is_unsupervised`);
/// making it optional in TOML is E53.
#[test]
fn accumulate_only_ew_cov_writes_n_eff_alone() {
    let input = tmp("bare-in.parquet");
    write_input(&input, 1000).unwrap();
    let output = tmp("bare-out.parquet");
    let state = tmp("bare.state");
    let toml = format!(
        r#"
input = "{}"
output = "{}"
save_state = "{}"
chunk_size = 256

[[specs]]
name = "g"
targets = ["x0"]
features = ["x0", "x1"]
clock = "t"
half_life = 50.0
gap_cap = 10.0
group = "group"

[specs.model]
type = "ew_cov"
stats = []
"#,
        toml_path(&input),
        toml_path(&output),
        toml_path(&state)
    );
    let cfg: RunConfig = toml::from_str(&toml).unwrap();
    let stats = run_config(&cfg, |_| Ok(())).unwrap();
    assert_eq!(stats.rows, 1000);

    let df = ParquetReader::new(std::fs::File::open(&output).unwrap())
        .finish()
        .unwrap();
    let fields = df
        .column("g")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series();
    let names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
    // Every row carries the two readiness fields beside `n_eff`
    // (docs/WARMUP-AND-CONVERGENCE.md §3); nothing else.
    assert_eq!(
        names,
        ["weight_sum", "settled_frac", "withheld_reason"],
        "an empty `stats` emits n_eff and the readiness fields, nothing else"
    );
    let n_eff = fields[0].f64().unwrap();
    assert!(
        n_eff.last().unwrap() > 0.0,
        "the model learned every row it saw"
    );

    let bank = Bank::load(&state, Some(&cfg.specs)).unwrap();
    let grams = bank.gram(0, None).unwrap();
    assert_eq!(grams.len(), 2, "one Gram per group");
    for g in &grams {
        assert_eq!(g.k, 2);
        assert!(g.n_eff > 0.0);
        assert!(g.comoments.iter().all(|v| v.is_finite()));
        assert!(g.cross_moments.is_empty(), "ew_cov has no targets");
    }

    // Without the line, the default list: three statistics over two columns.
    let default_toml = toml.replace("stats = []\n", "");
    assert!(
        default_toml.len() < toml.len(),
        "the test removed the stats line"
    );
    let default_cfg: RunConfig = toml::from_str(&default_toml).unwrap();
    let fields = online_polars::output_fields(&default_cfg.specs[0]);
    assert!(
        fields.len() > 1,
        "`stats` missing still means mean/std/corr: {fields:?}"
    );

    for p in [&input, &output, &state] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn rejects_a_config_with_no_specs() {
    let cfg: RunConfig = toml::from_str(
        r#"
input = "x.parquet"
output = "y.parquet"
specs = []
"#,
    )
    .unwrap();
    assert!(cfg.validate().unwrap_err().contains("no [[specs]]"));
}

#[test]
fn rejects_a_misspelt_key_with_its_line() {
    // A key the config has not got is an error naming it, where it is and
    // what the keys are -- not a default kept in silence. At every level a
    // TOML has: the run's keys, a spec's, and the model's.
    let good = r#"
input = "x.parquet"
output = "y.parquet"
chunk_size = 5
[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
half_life = 10
[specs.model]
type = "ewridge"
"#;
    assert!(toml::from_str::<RunConfig>(good).is_ok());
    let unknown = |text: String| toml::from_str::<RunConfig>(&text).unwrap_err().to_string();

    let err = unknown(good.replace("chunk_size = 5", "chunk_row = 5"));
    assert!(
        err.contains("line 4") && err.contains("unknown field `chunk_row`, expected one of"),
        "{err}"
    );
    let err = unknown(good.replace("half_life = 10", "halflfe = 10"));
    assert!(err.contains("unknown field `halflfe`"), "{err}");
    let err = unknown(good.replace("type = \"ewridge\"", "type = \"ewridge\"\nrigde = 0.1"));
    assert!(
        err.contains("unknown field `rigde`, expected one of `ridge`"),
        "{err}"
    );
}

/// A value of the wrong type inside a model is named by its spec and key:
/// serde reads the model into a buffer first, and TOML's error pointed at the
/// `[specs.model]` table alone (review 2026-10-06, PC11).
#[test]
fn a_model_value_of_the_wrong_type_is_named_by_its_key() {
    let dir = fresh_dir("pc11-toml");
    let cfg = dir.join("bank.toml");
    std::fs::write(
        &cfg,
        "input = \"x.parquet\"\noutput = \"y.parquet\"\n\n[[specs]]\nname = \"m\"\n\
         features = [\"x0\"]\nhalf_life = 10.0\n\n[specs.model]\ntype = \"micro\"\n\
         beta_mu = 3.0\neps = \"inf\"\n",
    )
    .unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_online"))
        .arg("--config")
        .arg(&cfg)
        .arg("--dry-run")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(
        err.contains("invalid type: string \"inf\", expected f64")
            && err.contains("at spec \"m\": model.eps"),
        "{err}"
    );
}

/// `gram_threads` is configuration, not state: a run saved under 8 threads
/// resumes under 2, or with the key left out, to the bit of the unbroken
/// run (review 6, D-1: the load compared it, as review 2026-09-26 F3 found
/// for marginal's `shards`, and refused the config).
#[test]
fn resume_under_another_gram_threads_continues_the_stream() {
    fn threads(cfg: &mut RunConfig, t: Option<usize>) {
        match &mut cfg.specs[0].model {
            online_polars::ModelKind::EwRidge { gram_threads, .. } => *gram_threads = t,
            other => panic!("the config is an ewridge: {other:?}"),
        }
    }
    let input = tmp("threads-in.parquet");
    write_input(&input, 1000).unwrap();
    let full_out = tmp("threads-full.parquet");
    run_config(&config(&input, &full_out, 100_000), |_| Ok(())).unwrap();
    let full = read_preds(&full_out).unwrap();
    let (half_a, half_b) = (tmp("threads-a.parquet"), tmp("threads-b.parquet"));
    {
        let df = ParquetReader::new(std::fs::File::open(&input).unwrap())
            .finish()
            .unwrap();
        let (mut a, mut b) = (df.slice(0, 500), df.slice(500, 500));
        ParquetWriter::new(std::fs::File::create(&half_a).unwrap())
            .finish(&mut a)
            .unwrap();
        ParquetWriter::new(std::fs::File::create(&half_b).unwrap())
            .finish(&mut b)
            .unwrap();
    }
    let state = tmp("threads.state");
    let out_a = tmp("threads-out-a.parquet");
    let mut cfg_a = config(&half_a, &out_a, 100_000);
    threads(&mut cfg_a, Some(8));
    cfg_a.save_state = Some(state.clone());
    run_config(&cfg_a, |_| Ok(())).unwrap();
    let out_b = tmp("threads-out-b.parquet");
    for t in [Some(2), None] {
        let mut cfg_b = config(&half_b, &out_b, 100_000);
        threads(&mut cfg_b, t);
        cfg_b.load_state = Some(state.clone());
        run_config(&cfg_b, |_| Ok(())).unwrap_or_else(|e| panic!("{t:?}: {e}"));
        let mut resumed = read_preds(&out_a).unwrap();
        resumed.extend(read_preds(&out_b).unwrap());
        assert_eq!(resumed, full, "{t:?} threads");
    }
    for p in [&input, &full_out, &half_a, &half_b, &state, &out_a, &out_b] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn resume_rejects_mismatched_specs() {
    let input = tmp("mismatch-in.parquet");
    write_input(&input, 200).unwrap();
    let state = tmp("mismatch.state");
    let out = tmp("mismatch-out.parquet");

    let mut cfg = config(&input, &out, 100_000);
    cfg.save_state = Some(state.clone());
    run_config(&cfg, |_| Ok(())).unwrap();

    let mut other = config(&input, &out, 100_000);
    other.load_state = Some(state.clone());
    other.specs[0].half_life = Some(online_polars::SpanList::One(online_polars::Span::Units(
        999.0,
    )));
    let err = run_config(&other, |_| Ok(())).unwrap_err().to_string();
    assert!(err.contains("do not match"), "{err}");

    for p in [&input, &state, &out] {
        let _ = std::fs::remove_file(p);
    }
}

/// The `online` binary over a config written to `dir`, with `args` after
/// `--config`: its exit status, stdout and stderr.
fn online(dir: &Path, input: &Path, output: &Path, args: &[&str]) -> (bool, String, String) {
    let (code, stdout, stderr) = online_with(dir, input, output, "", 50.0, args);
    (code == Some(0), stdout, stderr)
}

/// [`online`] with more top-level TOML keys (`top`, before `[[specs]]`) and
/// the spec's `half_life`, giving the exit code itself.
fn online_with(
    dir: &Path,
    input: &Path,
    output: &Path,
    top: &str,
    half_life: f64,
    args: &[&str],
) -> (Option<i32>, String, String) {
    let toml = format!(
        r#"
input = "{}"
output = "{}"
chunk_size = 100
{top}

[[specs]]
name = "ridge"
targets = ["y"]
features = ["x0", "x1"]
clock = "t"
half_life = {half_life:?}
gap_cap = 10.0
group = "group"

[specs.model]
type = "ewridge"
"#,
        toml_path(input),
        toml_path(output)
    );
    let cfg = dir.join("bank.toml");
    std::fs::write(&cfg, toml).unwrap();
    online_args(&[std::ffi::OsStr::new("--config"), cfg.as_os_str()], args)
}

/// The binary with `head` then `args`: its exit code, stdout and stderr.
fn online_args(head: &[&std::ffi::OsStr], args: &[&str]) -> (Option<i32>, String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_online"))
        .args(head)
        .args(args)
        .output()
        .unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn fresh_dir(name: &str) -> PathBuf {
    let dir = tmp(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Task 160, YB7: a `--save-state` that is a directory was found when the
/// state was saved, after the run had read the input and published the
/// output. It is refused before a row is read, naming the path.
#[test]
fn a_save_state_that_is_a_directory_is_refused_before_the_run() {
    let dir = fresh_dir("yb7");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 300).unwrap();
    let state = dir.join("a-directory");
    std::fs::create_dir_all(&state).unwrap();
    let (ok, _, err) = online(
        &dir,
        &input,
        &output,
        &["-q", "--save-state", state.to_str().unwrap()],
    );
    assert!(!ok, "the run succeeded");
    assert!(
        err.contains(&state.display().to_string()) && err.contains("is a directory"),
        "{err}"
    );
    assert!(
        !output.exists(),
        "the output was written before the refusal"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 160, YB12: the dry run said "config OK" for an input that is not
/// there. It reads what the run would read first, and fails naming it.
#[test]
fn a_dry_run_names_an_input_that_is_not_there() {
    let dir = fresh_dir("yb12");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 50).unwrap();
    let (ok, stdout, err) = online(&dir, &input, &output, &["--dry-run"]);
    assert!(ok, "{err}");
    assert!(stdout.contains("config OK"), "{stdout}");
    let missing = dir.join("nope.parquet");
    let (ok, stdout, err) = online(
        &dir,
        &input,
        &output,
        &["--dry-run", "--input", missing.to_str().unwrap()],
    );
    assert!(!ok, "{stdout}");
    assert!(!stdout.contains("config OK"), "{stdout}");
    assert!(err.contains(&missing.display().to_string()), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `path` without the column `drop`, written beside it as `to`.
fn without_column(path: &Path, drop: &str, to: &Path) {
    let mut df = ParquetReader::new(std::fs::File::open(path).unwrap())
        .finish()
        .unwrap()
        .drop(drop)
        .unwrap();
    ParquetWriter::new(std::fs::File::create(to).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// Review round 4, SF2: the dry run said "config OK" for three runs that
/// fail at their first step -- a `--load-state` state that is not there, one
/// saved from other specs, and an input, or a `keep_columns`, without a
/// column a spec reads. It opens the bank as the run would and runs it on a
/// frame of no rows of the input's schema, so it refuses each as the run
/// does, and still reads no row.
#[test]
fn a_dry_run_refuses_what_the_run_refuses_at_its_first_step() {
    let dir = fresh_dir("sf2");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 60).unwrap();
    let (state, other) = (dir.join("bank.state"), dir.join("other.state"));
    for (path, half_life) in [(&state, 50.0), (&other, 999.0)] {
        let save = ["-q", "--save-state", path.to_str().unwrap()];
        let (code, _, err) = online_with(&dir, &input, &output, "", half_life, &save);
        assert_eq!(code, Some(0), "{err}");
    }
    let s = state.to_str().unwrap();
    for args in [
        &["--dry-run"][..],
        &["--dry-run", "--load-state", s],
        &["--dry-run", "--load-state", s, "--predict"],
    ] {
        let (code, stdout, err) = online_with(&dir, &input, &output, "", 50.0, args);
        assert_eq!(code, Some(0), "{args:?}: {err}");
        assert!(stdout.contains("config OK"), "{args:?}: {stdout}");
    }
    let missing = dir.join("missing.state");
    let narrow = dir.join("narrow.parquet");
    without_column(&input, "x1", &narrow);
    let keep = r#"keep_columns = ["group", "t", "x0", "y"]"#;
    let missing_text = missing.display().to_string();
    let cases: [(&str, &str, Vec<&str>, Vec<&str>); 4] = [
        (
            "a state that is not there",
            "",
            vec!["--load-state", missing.to_str().unwrap()],
            vec!["loading state", missing_text.as_str()],
        ),
        (
            "a state saved from other specs",
            "",
            vec!["--load-state", other.to_str().unwrap()],
            vec!["do not match"],
        ),
        (
            "an input without a feature",
            "",
            vec!["--input", narrow.to_str().unwrap()],
            vec!["\"x1\"", "not found"],
        ),
        (
            "keep_columns without a feature",
            keep,
            vec![],
            vec!["\"x1\"", "not found"],
        ),
    ];
    for (label, top, args, want) in cases {
        let mut dry = vec!["--dry-run"];
        dry.extend_from_slice(&args);
        let (code, stdout, err) = online_with(&dir, &input, &output, top, 50.0, &dry);
        assert_eq!(code, Some(1), "{label}: {stdout}{err}");
        assert!(!stdout.contains("config OK"), "{label}: {stdout}");
        // And the run refuses it, for the same reason.
        let mut run = vec!["-q"];
        run.extend_from_slice(&args);
        let (run_code, _, run_err) = online_with(&dir, &input, &output, top, 50.0, &run);
        assert_eq!(run_code, Some(1), "{label}: the run");
        for w in &want {
            assert!(err.contains(w), "{label}: {w:?} not in the dry run's {err}");
            assert!(
                run_err.contains(w),
                "{label}: {w:?} not in the run's {run_err}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Review round 4, SF6: a deployment scripts on the exit status, so it is
/// pinned. 0 for a run that finished; 1 for every refusal and every error
/// of a run -- a config that does not parse or is not there, an input that
/// is not there, a chunk the bank refuses -- with `online: <message>` on
/// stderr; and 2 for a command line clap refuses, a usage error.
#[test]
fn the_exit_status_is_0_for_a_run_1_for_a_refusal_and_2_for_usage() {
    let dir = fresh_dir("sf6");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 60).unwrap();
    let (code, stdout, err) = online_with(&dir, &input, &output, "", 50.0, &["-q"]);
    assert_eq!(code, Some(0), "{err}");
    assert!(stdout.contains("wrote 60 rows"), "{stdout}");

    let narrow = dir.join("narrow.parquet");
    without_column(&input, "x1", &narrow);
    let bad_toml = dir.join("bad.toml");
    std::fs::write(&bad_toml, "input = \n").unwrap();
    let nowhere = dir.join("nowhere.toml");
    let missing = dir.join("missing.parquet");
    let config = |p: &Path| vec![std::ffi::OsString::from("--config"), p.into()];
    let refusals: [(&str, Vec<std::ffi::OsString>, Vec<&str>); 5] = [
        ("a config that does not parse", config(&bad_toml), vec![]),
        ("a config that is not there", config(&nowhere), vec![]),
        (
            "an input that is not there",
            config(&dir.join("bank.toml")),
            vec!["--input", missing.to_str().unwrap()],
        ),
        (
            "a chunk the bank refuses",
            config(&dir.join("bank.toml")),
            vec!["-q", "--input", narrow.to_str().unwrap()],
        ),
        (
            "a config the binary refuses",
            config(&dir.join("bank.toml")),
            vec!["--no-output"],
        ),
    ];
    for (label, head, args) in refusals {
        let head: Vec<&std::ffi::OsStr> = head.iter().map(|s| s.as_os_str()).collect();
        let (code, _, err) = online_args(&head, &args);
        assert_eq!(code, Some(1), "{label}: {err}");
        assert!(err.starts_with("online: "), "{label}: {err}");
    }
    let usage: [(&str, &[&str]); 4] = [
        ("no --config", &[]),
        ("an unknown flag", &["--no-such-flag"]),
        ("a format clap cannot parse", &["--input-format", "xlsx"]),
        (
            "two flags that conflict",
            &["--output", "o.parquet", "--no-output"],
        ),
    ];
    let cfg = dir.join("bank.toml");
    for (label, args) in usage {
        let head: Vec<&std::ffi::OsStr> = if label == "no --config" {
            vec![]
        } else {
            vec![std::ffi::OsStr::new("--config"), cfg.as_os_str()]
        };
        let (code, _, err) = online_args(&head, args);
        assert_eq!(code, Some(2), "{label}: {err}");
        // clap's own voice, not the binary's `online: `.
        assert!(err.starts_with("error: "), "{label}: {err}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 160, YB13: `--no-output --predict` can never run -- a scoring run's
/// product is its output -- and the refusal asked for `closed_groups` or
/// `save_state`, both of which `--predict` refuses. It names `--predict`.
#[test]
fn no_output_with_predict_is_refused_naming_predict() {
    let dir = fresh_dir("yb13");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 300).unwrap();
    let state = dir.join("bank.state");
    let (ok, _, err) = online(
        &dir,
        &input,
        &output,
        &["-q", "--save-state", state.to_str().unwrap()],
    );
    assert!(ok, "{err}");
    for extra in [&[][..], &["--dry-run"][..]] {
        let mut args = vec![
            "--no-output",
            "--predict",
            "--load-state",
            state.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let (ok, _, err) = online(&dir, &input, &output, &args);
        assert!(!ok, "{extra:?}");
        assert!(
            err.contains("predict") && err.contains("its output is its only product"),
            "{extra:?}: {err}"
        );
        assert!(!err.contains("closed_groups"), "{extra:?}: {err}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The first `n` rows of `path`, written to `to`.
fn head_of(path: &Path, n: usize, to: &Path) {
    let mut df = ParquetReader::new(std::fs::File::open(path).unwrap())
        .finish()
        .unwrap()
        .slice(0, n);
    ParquetWriter::new(std::fs::File::create(to).unwrap())
        .finish(&mut df)
        .unwrap();
}

fn read_frame(path: &Path) -> DataFrame {
    ParquetReader::new(std::fs::File::open(path).unwrap())
        .finish()
        .unwrap()
}

/// Task 196 (N2, N15): the chunk size is Polars' `chunk_size`, on the flag
/// and the TOML key, and the state to start from is `--load-state`, as the
/// key `load_state` and `--save-state` are. The old flags are refused as a
/// usage error naming the new ones, and the old key as the config's own
/// refusal does, naming the new one.
#[test]
fn the_renamed_flags_and_key_are_refused_naming_the_new_ones() {
    let dir = fresh_dir("t196-names");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 300).unwrap();
    let state = dir.join("bank.state");
    let state_arg = state.to_str().unwrap();
    let (code, stdout, err) = online_with(
        &dir,
        &input,
        &output,
        "",
        50.0,
        &["--chunk-size", "70", "--save-state", state_arg, "--dry-run"],
    );
    assert_eq!(code, Some(0), "{err}");
    assert!(stdout.contains("chunk_size: 70"), "{stdout}");
    let (code, _, err) = online_with(
        &dir,
        &input,
        &output,
        "",
        50.0,
        &["-q", "--chunk-size", "70", "--save-state", state_arg],
    );
    assert_eq!(code, Some(0), "{err}");
    let (code, _, err) = online_with(
        &dir,
        &input,
        &output,
        "",
        50.0,
        &["-q", "--load-state", state_arg, "--predict"],
    );
    assert_eq!(code, Some(0), "{err}");
    let old: [(&[&str], &str); 2] = [
        (
            &["--chunk-rows", "70"],
            "--chunk-rows was renamed --chunk-size",
        ),
        (
            &["--resume", state_arg],
            "--resume was renamed --load-state",
        ),
    ];
    for (args, msg) in old {
        let (code, _, err) = online_with(&dir, &input, &output, "", 50.0, args);
        assert_eq!(code, Some(2), "{args:?}: {err}");
        assert!(err.contains(msg), "{args:?}: {err}");
    }
    // The key: the config written with the old one, the run refused.
    let toml = std::fs::read_to_string(dir.join("bank.toml"))
        .unwrap()
        .replace("chunk_size = ", "chunk_rows = ");
    let old_key = dir.join("old.toml");
    std::fs::write(&old_key, toml).unwrap();
    let (code, _, err) = online_args(
        &[std::ffi::OsStr::new("--config"), old_key.as_os_str()],
        &["-q"],
    );
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("unknown field `chunk_rows`")
            && err.contains("chunk_rows was renamed chunk_size"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 196 (N26, review AP22): `--skip-learned` resumes a saved bank on
/// input that overlaps it, as `ModelBank.skip_learned` does in Python: the
/// rows at or before each group's last learned clock are dropped, so each
/// row is learned once. Without it the overlap steps the clock back and the
/// run is refused. Two interleaved groups, the state saved after 200 rows,
/// the whole 300-row file run again: the output is rows 200 to 299, and
/// their predictions are the unbroken run's, to the bit.
#[test]
fn skip_learned_resumes_on_overlapping_input() {
    let dir = fresh_dir("t196-skip");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 300).unwrap();
    let head = dir.join("head.parquet");
    head_of(&input, 200, &head);
    let state = dir.join("bank.state");
    let state_arg = state.to_str().unwrap();
    let full = dir.join("full.parquet");
    let (code, _, err) = online_with(&dir, &input, &full, "", 50.0, &["-q", "--chunk-size", "64"]);
    assert_eq!(code, Some(0), "{err}");
    let (code, _, err) = online_with(
        &dir,
        &head,
        &dir.join("head-out.parquet"),
        "",
        50.0,
        &["-q", "--save-state", state_arg],
    );
    assert_eq!(code, Some(0), "{err}");

    // Without it: the overlap is a step back, refused.
    let (code, _, err) = online_with(
        &dir,
        &input,
        &output,
        "",
        50.0,
        &["-q", "--load-state", state_arg],
    );
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("restart_after_step_back"), "{err}");

    for chunk in ["64", "300"] {
        let (code, _, err) = online_with(
            &dir,
            &input,
            &output,
            "",
            50.0,
            &[
                "-q",
                "--load-state",
                state_arg,
                "--skip-learned",
                "--chunk-size",
                chunk,
            ],
        );
        assert_eq!(code, Some(0), "{chunk}: {err}");
        // The numbers, not `coef`, which is written on each group's last
        // row of a chunk, and the chunks differ.
        let fields = |df: DataFrame| -> DataFrame {
            let ridge = df.column("ridge").unwrap().struct_().unwrap().clone();
            let mut cols: Vec<Column> = ["group", "t", "x0", "x1", "y"]
                .iter()
                .map(|c| df.column(c).unwrap().clone())
                .collect();
            for f in ["pred_y", "resid_y", "weight_sum"] {
                cols.push(ridge.field_by_name(f).unwrap().into());
            }
            DataFrame::new(df.height(), cols).unwrap()
        };
        let got = fields(read_frame(&output));
        let want = fields(read_frame(&full).slice(200, 100));
        assert_eq!(got.height(), 100, "{chunk}");
        assert!(got.equals_missing(&want), "{chunk}: {got} vs {want}");
    }

    // A fresh bank has learned nothing to skip.
    let (code, _, err) = online_with(&dir, &input, &output, "", 50.0, &["-q", "--skip-learned"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("skip_learned") && err.contains("load_state"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Rows `from` on of `path`, the clock of row `nan_at` among them NaN,
/// written as `to`.
fn tail_with_nan_clock(path: &Path, from: usize, nan_at: usize, to: &Path) {
    let mut df = read_frame(path).slice(from as i64, usize::MAX);
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .enumerate()
        .map(|(i, v)| if i == nan_at { f64::NAN } else { v })
        .collect();
    df.with_column(Column::new("t".into(), t)).unwrap();
    ParquetWriter::new(std::fs::File::create(to).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// Review round 5 (C3): `--skip-learned` read a clock that is not a number
/// as "not after the group's last learned clock", and dropped the row in
/// silence, where the bank refuses such a row by its position and
/// `ModelBank.skip_learned` keeps it for the bank to refuse. The rows past
/// a saved state, the sixth with a NaN clock: exit 1 naming row 5, with
/// `--skip-learned` as without it, and no output written.
#[test]
fn skip_learned_keeps_a_nan_clock_for_the_bank_to_refuse() {
    let dir = fresh_dir("r5-c3-nan-clock");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    write_input(&input, 40).unwrap();
    let head = dir.join("head.parquet");
    head_of(&input, 20, &head);
    let state = dir.join("bank.state");
    let state_arg = state.to_str().unwrap();
    let (code, _, err) = online_with(
        &dir,
        &head,
        &dir.join("head-out.parquet"),
        "",
        50.0,
        &["-q", "--save-state", state_arg],
    );
    assert_eq!(code, Some(0), "{err}");
    let rest = dir.join("rest.parquet");
    tail_with_nan_clock(&input, 20, 5, &rest);
    for args in [
        &["-q", "--load-state", state_arg][..],
        &["-q", "--load-state", state_arg, "--skip-learned"][..],
    ] {
        let (code, _, err) = online_with(&dir, &rest, &output, "", 50.0, args);
        assert_eq!(code, Some(1), "{args:?}: {err}");
        assert!(
            err.contains("clock column \"t\" has a null/non-finite value at row 5"),
            "{args:?}: {err}"
        );
        assert!(!output.exists(), "{args:?}: an output was written");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Task 198 (D5): a formula target's written form -- its tree in the TOML
/// file and in a saved state -- is labelled unstable. Under
/// `POLARS_ONLINE_WARN_UNSTABLE=1` the command line says so on stderr, as
/// Polars' `POLARS_WARN_UNSTABLE` does; without it, or for a config with no
/// formula target, it says nothing.
#[test]
fn a_formula_target_is_labelled_unstable_under_the_variable() {
    let dir = fresh_dir("unstable");
    let (input, output) = (dir.join("in.parquet"), dir.join("out.parquet"));
    let n = 40;
    let mut df = df!(
        "t" => (0..n).map(f64::from).collect::<Vec<_>>(),
        "mid" => (0..n).map(|i| 100.0 + 0.1 * f64::from(i % 7)).collect::<Vec<_>>(),
        "x0" => (0..n).map(|i| f64::from(i % 5) - 2.0).collect::<Vec<_>>(),
        "y" => (0..n).map(|i| f64::from(i % 3)).collect::<Vec<_>>()
    )
    .unwrap();
    ParquetWriter::new(std::fs::File::create(&input).unwrap())
        .finish(&mut df)
        .unwrap();
    let formula = r#"[{ name = "fwd", formula = ["-", ["rewm_mean", ["col", "mid"], { half_life = 5.0, window_size = 10.0 }], ["col", "mid"]] }]"#;
    let run = |targets: &str, var: Option<&str>| {
        let toml = format!(
            "input = \"{}\"\noutput = \"{}\"\n\n[[specs]]\nname = \"edge\"\ntargets = {targets}\n\
             features = [\"x0\"]\nclock = \"t\"\ngap_cap = 50.0\nhalf_life = 40.0\nembargo = 12.0\n\
             [specs.model]\ntype = \"ewridge\"\n",
            toml_path(&input),
            toml_path(&output)
        );
        let cfg = dir.join("edge.toml");
        std::fs::write(&cfg, toml).unwrap();
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_online"));
        cmd.args(["--config", cfg.to_str().unwrap(), "--dry-run"]);
        cmd.env_remove("POLARS_ONLINE_WARN_UNSTABLE");
        if let Some(v) = var {
            cmd.env("POLARS_ONLINE_WARN_UNSTABLE", v);
        }
        let out = cmd.output().unwrap();
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{err}");
        err
    };
    let warned = run(formula, Some("1"));
    assert!(
        warned.contains("formula target's written form") && warned.contains("unstable"),
        "{warned}"
    );
    for (targets, var) in [
        (formula, None),
        (formula, Some("0")),
        ("[\"y\"]", Some("1")),
    ] {
        let quiet = run(targets, var);
        assert!(!quiet.contains("unstable"), "{targets} {var:?}: {quiet}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// docs/PLAN.md task 116 (H; docs/WARMUP-AND-CONVERGENCE.md §7.9): after
/// "wrote N rows", one line per spec with something to say -- the groups
/// whose last row was withheld, counted by `withheld_reason`, and the groups
/// whose smallest data share is below 0.5 -- and none for a spec with
/// neither. A `min_weight` no stream reaches withholds both groups' every
/// row; a duplicated feature under a ridge of 0.5 reads `a / (2a + 0.5)`,
/// 0.29 here, in both groups.
#[test]
fn the_run_closes_with_a_line_per_spec_that_is_not_ready() {
    let dir = fresh_dir("t116h");
    let (raw, input, output) = (
        dir.join("raw.parquet"),
        dir.join("in.parquet"),
        dir.join("out.parquet"),
    );
    write_input(&raw, 200).unwrap();
    let mut df = ParquetReader::new(std::fs::File::open(&raw).unwrap())
        .finish()
        .unwrap();
    let x2 = df.column("x0").unwrap().clone().with_name("x2".into());
    df.with_column(x2).unwrap();
    ParquetWriter::new(std::fs::File::create(&input).unwrap())
        .finish(&mut df)
        .unwrap();
    let spec = |name: &str, features: &str, model: &str, top: &str| {
        format!(
            r#"
[[specs]]
name = "{name}"
targets = ["y"]
features = {features}
clock = "t"
half_life = 50.0
gap_cap = 10.0
group = "group"
{top}

[specs.model]
type = "ewridge"
{model}
"#
        )
    };
    let toml = format!(
        "input = \"{}\"\noutput = \"{}\"\nchunk_size = 64\n{}{}{}",
        toml_path(&input),
        toml_path(&output),
        spec("plain", r#"["x0", "x1"]"#, "", ""),
        spec("held", r#"["x0", "x1"]"#, "", "min_weight = 1e9"),
        spec("dup", r#"["x0", "x1", "x2"]"#, "ridge = 0.5", ""),
    );
    let cfg = dir.join("bank.toml");
    std::fs::write(&cfg, toml).unwrap();
    let head = [std::ffi::OsStr::new("--config"), cfg.as_os_str()];
    let (code, stdout, err) = online_args(&head, &["-q"]);
    assert_eq!(code, Some(0), "{err}");
    let lines: Vec<&str> = stdout.lines().collect();
    let wrote = lines
        .iter()
        .position(|l| l.starts_with("wrote 200 rows"))
        .unwrap_or_else(|| panic!("{stdout}"));
    assert_eq!(
        &lines[wrote + 1..],
        [
            r#"spec "held": 2 groups whose last row was withheld (below_min_weight 2)"#,
            r#"spec "dup": 2 groups with min_support_coef < 0.5"#,
        ],
        "{stdout}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
