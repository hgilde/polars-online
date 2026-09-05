//! CLI-level tests: streaming a parquet file through a TOML config, chunk
//! invariance across chunk_rows, and resume-from-state (docs/PLAN.md task 15).

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

fn config(input: &Path, output: &Path, chunk_rows: usize) -> RunConfig {
    let toml = format!(
        r#"
input = "{}"
output = "{}"
chunk_rows = {chunk_rows}

[[specs]]
name = "ridge"
targets = ["y"]
features = ["x0", "x1"]
clock = "t"
halflife = 50.0
max_dclock = 10.0
weight = "w"
group = "group"
min_periods = 5.0

[specs.model]
type = "ew_ridge"
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
/// groups were not a multiple of `chunk_rows`.
#[test]
fn row_groups_need_not_align_with_chunk_rows() {
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
chunk_rows = 256

[[specs]]
name = "g"
targets = ["x0"]
features = ["x0", "x1"]
clock = "t"
halflife = 50.0
max_dclock = 10.0
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
    assert_eq!(
        names,
        ["n_eff"],
        "an empty `stats` emits n_eff and nothing else"
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
chunk_rows = 5
[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
halflife = 10
[specs.model]
type = "ew_ridge"
"#;
    assert!(toml::from_str::<RunConfig>(good).is_ok());
    let unknown = |text: String| toml::from_str::<RunConfig>(&text).unwrap_err().to_string();

    let err = unknown(good.replace("chunk_rows = 5", "chunk_row = 5"));
    assert!(
        err.contains("line 4") && err.contains("unknown field `chunk_row`, expected one of"),
        "{err}"
    );
    let err = unknown(good.replace("halflife = 10", "halflfe = 10"));
    assert!(err.contains("unknown field `halflfe`"), "{err}");
    let err = unknown(good.replace("type = \"ew_ridge\"", "type = \"ew_ridge\"\nrigde = 0.1"));
    assert!(
        err.contains("unknown field `rigde`, expected one of `ridge`"),
        "{err}"
    );
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
    other.specs[0].halflife = Some(online_polars::FloatOrList::Float(online_polars::Num(999.0)));
    let err = run_config(&other, |_| Ok(())).unwrap_err().to_string();
    assert!(err.contains("do not match"), "{err}");

    for p in [&input, &state, &out] {
        let _ = std::fs::remove_file(p);
    }
}
