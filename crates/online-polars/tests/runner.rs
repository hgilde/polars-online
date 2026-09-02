//! Runner-level tests (docs/ENHANCEMENTS.md E32): the same numbers whichever
//! format the rows come in or go out in, frames from a plan or from an
//! iterator, files or callbacks out, and the edges -- empty streams, explicit
//! formats, a run that fails after an earlier one succeeded.

use std::path::{Path, PathBuf};

use online_polars::{
    Bank, Format, Input, Output, RunConfig, RunOptions, RunStats, run, run_config, run_config_on,
};
use polars::prelude::*;

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "polars-online-runner-test-{}-{name}",
        std::process::id()
    ));
    p
}

/// Deterministic two-group stream.
fn stream(n: usize) -> DataFrame {
    let mut s = 77u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut group, mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs();
        t.push(clocks[g]);
        let (a, b) = (lcg(), lcg());
        x0.push(a);
        x1.push(b);
        y.push(if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg());
    }
    df!("group" => group, "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// `df` written to `path` in `format` with polars' own writer.
fn write(path: &Path, format: Format, df: &DataFrame) {
    let mut df = df.clone();
    let file = std::fs::File::create(path).unwrap();
    match format {
        Format::Parquet => ParquetWriter::new(file).finish(&mut df).map(|_| ()),
        Format::Ipc => IpcWriter::new(file).finish(&mut df),
        Format::Csv => CsvWriter::new(file).finish(&mut df),
        Format::Ndjson => JsonWriter::new(file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut df),
    }
    .unwrap();
}

/// `path` read back with the runner's own scan.
fn read(path: &Path, format: Format) -> DataFrame {
    format.scan(path).unwrap().collect().unwrap()
}

fn names(df: &DataFrame) -> Vec<&str> {
    df.get_column_names()
        .into_iter()
        .map(|s| s.as_str())
        .collect()
}

/// The extension each format is told from.
fn ext(format: Format) -> &'static str {
    match format {
        Format::Parquet => "parquet",
        Format::Ipc => "arrow",
        Format::Csv => "csv",
        Format::Ndjson => "jsonl",
    }
}

fn spec() -> online_polars::Spec {
    spec_with_coef_every(1)
}

fn spec_with_coef_every(coef_every: usize) -> online_polars::Spec {
    serde_json::from_str(&format!(
        r#"{{
            "name": "ridge",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": 50.0,
            "max_dclock": 10.0,
            "group": "group",
            "min_periods": 5.0,
            "coef_every": {coef_every}
        }}"#
    ))
    .unwrap()
}

fn config(input: &Path, output: &Path) -> RunConfig {
    RunConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        input_format: None,
        output_format: None,
        chunk_rows: 64,
        load_state: None,
        save_state: None,
        keep_columns: vec![],
        predict: false,
        specs: vec![spec()],
    }
}

fn no_progress(_: RunStats) -> PolarsResult<()> {
    Ok(())
}

/// `pred_y` from an output frame: the struct field, or the flattened CSV
/// column.
fn preds(df: &DataFrame) -> Vec<Option<f64>> {
    let s = match df.column("ridge") {
        Ok(c) => c.struct_().unwrap().field_by_name("pred_y").unwrap(),
        Err(_) => df
            .column("ridge.pred_y")
            .unwrap()
            .as_materialized_series()
            .clone(),
    };
    s.f64().unwrap().iter().collect()
}

/// What the bank says on its own, the oracle for every path through the
/// runner.
fn bank_preds(df: &DataFrame) -> Vec<Option<f64>> {
    let mut bank = Bank::new(vec![spec()]).unwrap();
    let cols = bank.fit_predict(df).unwrap();
    preds(&DataFrame::new(df.height(), cols).unwrap())
}

fn cleanup(paths: &[PathBuf]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn every_input_format_gives_the_banks_numbers() {
    let df = stream(500);
    let want = bank_preds(&df);
    let mut files = vec![];
    for format in Format::ALL {
        let input = tmp(&format!("in-{}.{}", format.name(), ext(format)));
        let output = tmp(&format!("in-{}-out.parquet", format.name()));
        write(&input, format, &df);
        let stats = run_config(&config(&input, &output), no_progress).unwrap();
        assert_eq!((stats.rows, stats.chunks), (500, 8), "{format:?}");
        let got = read(&output, Format::Parquet);
        assert_eq!(preds(&got), want, "{format:?} input");
        // The input columns come through as they were.
        assert_eq!(got.drop("ridge").unwrap(), df, "{format:?} input");
        files.extend([input, output]);
    }
    cleanup(&files);
}

#[test]
fn every_output_format_carries_the_banks_columns() {
    let df = stream(500);
    let want = bank_preds(&df);
    let input = tmp("out-in.parquet");
    write(&input, Format::Parquet, &df);
    let mut files = vec![input.clone()];
    for format in Format::ALL {
        let output = tmp(&format!("out-{}.{}", format.name(), ext(format)));
        run_config(&config(&input, &output), no_progress).unwrap();
        let got = read(&output, format);
        assert_eq!(got.height(), 500, "{format:?}");
        assert_eq!(preds(&got), want, "{format:?} output");
        match format {
            // CSV has no nested values: the struct is flattened to
            // `<spec>.<field>` columns and the list field is a JSON string.
            Format::Csv => {
                assert_eq!(
                    names(&got),
                    [
                        "group",
                        "t",
                        "x0",
                        "x1",
                        "y",
                        "ridge.pred_y",
                        "ridge.resid_y",
                        "ridge.n_eff",
                        "ridge.coef"
                    ]
                );
                let coef = got.column("ridge.coef").unwrap();
                assert_eq!(coef.dtype(), &DataType::String);
                let last = coef.str().unwrap().last().unwrap();
                assert!(last.starts_with('[') && last.ends_with(']'), "{last}");
                // Intercept and two features.
                assert_eq!(last.split(',').count(), 3, "{last}");
            }
            _ => {
                let ridge = got.column("ridge").unwrap();
                assert!(matches!(ridge.dtype(), DataType::Struct(_)), "{format:?}");
                let coef = ridge.struct_().unwrap().field_by_name("coef").unwrap();
                assert_eq!(coef.dtype(), &DataType::List(Box::new(DataType::Float64)));
            }
        }
        files.push(output);
    }
    cleanup(&files);
}

#[test]
fn csv_lists_read_back_as_lists() {
    let df = stream(200);
    let input = tmp("csv-list-in.parquet");
    let output = tmp("csv-list-out.csv");
    write(&input, Format::Parquet, &df);
    // A snapshot every 7 rows, so the rows between are null and must come
    // back null (an empty CSV field) rather than as an empty list.
    let mut cfg = config(&input, &output);
    cfg.specs = vec![spec_with_coef_every(7)];
    run_config(&cfg, no_progress).unwrap();
    let got = read(&output, Format::Csv);
    let nulls = got.column("ridge.coef").unwrap().null_count();
    assert!(nulls > 100 && nulls < 200, "{nulls}");
    // The documented way back is `str.json_decode(pl.List(pl.Float64))`,
    // polars' JSON parser (held bit-exact in tests/test_runner.py); here,
    // the same parse by hand with the standard library's correctly rounded
    // `f64::from_str`. (serde_json's default float parsing is best-effort
    // and lands an ulp off on some of these.)
    let coef = got.column("ridge.coef").unwrap();
    let parsed: Vec<Option<Vec<f64>>> = coef
        .str()
        .unwrap()
        .iter()
        .map(|s| {
            s.map(|s| {
                s.trim_matches(['[', ']'])
                    .split(',')
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect()
            })
        })
        .collect();
    // The oracle sees the runner's chunks: `coef` is also snapshotted on
    // each chunk's last row (the documented exception to chunk invariance).
    let mut bank = Bank::new(vec![spec_with_coef_every(7)]).unwrap();
    let mut want: Vec<Option<Vec<f64>>> = vec![];
    for start in (0..df.height()).step_by(64) {
        let cols = bank.fit_predict(&df.slice(start as i64, 64)).unwrap();
        let coef = cols[0].struct_().unwrap().field_by_name("coef").unwrap();
        want.extend(
            coef.list()
                .unwrap()
                .amortized_iter()
                .map(|s| s.map(|s| s.as_ref().f64().unwrap().iter().flatten().collect())),
        );
    }
    assert_eq!(parsed, want);
    cleanup(&[input, output]);
}

#[test]
fn frames_from_an_iterator_equal_the_plan() {
    let df = stream(700);
    let output_plan = tmp("frames-plan.parquet");
    let output_iter = tmp("frames-iter.parquet");
    let cfg = config(Path::new(""), &output_plan);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    // Uneven frames, and a size `chunk_rows` says nothing about.
    let frames = (0..700)
        .step_by(150)
        .map(move |i| Ok(df.slice(i as i64, 150)));
    let cfg = config(Path::new(""), &output_iter);
    let stats = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames),
            schema: stream(0).schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    assert_eq!((stats.rows, stats.chunks), (700, 5));
    assert_eq!(
        read(&output_iter, Format::Parquet),
        read(&output_plan, Format::Parquet)
    );
    cleanup(&[output_plan, output_iter]);
}

#[test]
fn keep_columns_applies_to_frames_too() {
    let df = stream(100)
        .with_column(Column::new("extra".into(), vec![1i32; 100]))
        .unwrap()
        .clone();
    let output = tmp("keep-frames.parquet");
    let mut cfg = config(Path::new(""), &output);
    cfg.keep_columns = ["group", "t", "x0", "x1", "y"].map(String::from).to_vec();
    let frames = std::iter::once(Ok(df.clone()));
    run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    let got = read(&output, Format::Parquet);
    assert_eq!(names(&got), ["group", "t", "x0", "x1", "y", "ridge"]);
    cleanup(&[output]);
}

#[test]
fn frames_to_a_callback_equal_the_file() {
    let df = stream(300);
    let output = tmp("callback.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let from_file = read(&output, Format::Parquet);

    let mut bank = Bank::new(vec![spec()]).unwrap();
    let mut got: Vec<DataFrame> = vec![];
    let mut sink = |frame: DataFrame| {
        got.push(frame);
        Ok(())
    };
    let opts = RunOptions {
        chunk_rows: 64,
        predict: false,
    };
    let stats = run(
        &mut bank,
        Input::Lazy(df.lazy()),
        Output::Batches(&mut sink),
        opts,
        no_progress,
    )
    .unwrap();
    assert_eq!((stats.rows, stats.chunks), (300, 5));
    assert_eq!(got.len(), 5);
    let whole = got
        .into_iter()
        .reduce(|a, b| a.vstack(&b).unwrap())
        .unwrap();
    assert_eq!(whole, from_file);
    cleanup(&[output]);
}

#[test]
fn an_empty_stream_writes_an_empty_output_in_every_format() {
    let empty = stream(0);
    let input = tmp("empty-in.parquet");
    write(&input, Format::Parquet, &empty);
    let mut files = vec![input.clone()];
    for format in Format::ALL {
        let output = tmp(&format!("empty-out.{}", ext(format)));
        let stats = run_config(&config(&input, &output), no_progress).unwrap();
        assert_eq!((stats.rows, stats.chunks), (0, 0));
        match format {
            // No rows, no header, no footer: an empty file.
            Format::Ndjson => assert_eq!(std::fs::metadata(&output).unwrap().len(), 0),
            _ => {
                let got = read(&output, format);
                assert_eq!(got.height(), 0, "{format:?}");
                let last = names(&got).last().unwrap().to_string();
                assert!(last.starts_with("ridge"), "{format:?}: {last}");
            }
        }
        files.push(output);
    }
    // And from frames that turn out empty.
    let output = tmp("empty-frames.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(std::iter::empty()),
            schema: empty.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap();
    assert_eq!(
        names(&read(&output, Format::Parquet)),
        ["group", "t", "x0", "x1", "y", "ridge"]
    );
    files.push(output);
    cleanup(&files);
}

#[test]
fn an_explicit_format_overrides_the_extension() {
    let df = stream(100);
    let input = tmp("explicit-in.dat");
    let output = tmp("explicit-out.bin");
    write(&input, Format::Ipc, &df);
    let mut cfg = config(&input, &output);
    assert!(
        run_config(&cfg, no_progress)
            .unwrap_err()
            .to_string()
            .contains("cannot tell the format of"),
        "an unknown extension names the fix"
    );
    cfg.input_format = Some(Format::Ipc);
    cfg.output_format = Some(Format::Ndjson);
    run_config(&cfg, no_progress).unwrap();
    assert_eq!(preds(&read(&output, Format::Ndjson)), bank_preds(&df));
    cleanup(&[input, output]);
}

#[test]
fn a_failed_run_leaves_the_previous_output_where_it_was() {
    let df = stream(200);
    let output = tmp("failed-run.parquet");
    let cfg = config(Path::new(""), &output);
    run_config_on(&cfg, Input::Lazy(df.clone().lazy()), no_progress).unwrap();
    let before = std::fs::read(&output).unwrap();

    // The source fails halfway.
    let frames = vec![
        Ok(df.slice(0, 100)),
        Err(polars_err!(ComputeError: "the source went away")),
    ];
    let err = run_config_on(
        &cfg,
        Input::Batches {
            frames: Box::new(frames.into_iter()),
            schema: df.schema().as_ref().clone(),
        },
        no_progress,
    )
    .unwrap_err();
    assert!(err.to_string().contains("the source went away"), "{err}");
    assert_eq!(std::fs::read(&output).unwrap(), before);

    // And so does a progress callback that gives up.
    let err = run_config_on(&cfg, Input::Lazy(df.lazy()), |s| {
        if s.chunks == 2 {
            polars_bail!(ComputeError: "stop here");
        }
        Ok(())
    })
    .unwrap_err();
    assert!(err.to_string().contains("stop here"), "{err}");
    assert_eq!(std::fs::read(&output).unwrap(), before);
    // No temporary is left behind.
    let dir = output.parent().unwrap();
    let stem = output.file_name().unwrap().to_str().unwrap();
    let leftovers: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(stem) && n != stem)
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    cleanup(&[output]);
}

#[test]
fn progress_reports_every_chunk() {
    let df = stream(250);
    let output = tmp("progress.parquet");
    let cfg = config(Path::new(""), &output);
    let mut seen = vec![];
    run_config_on(&cfg, Input::Lazy(df.lazy()), |s| {
        seen.push((s.rows, s.chunks));
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, [(64, 1), (128, 2), (192, 3), (250, 4)]);
    cleanup(&[output]);
}
