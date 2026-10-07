//! `online` — run a model bank over a stream of rows, config from TOML
//! (docs/PLAN.md §11 task 15). Reads and writes parquet, ipc, csv and ndjson,
//! each told from its extension or named with `--input-format` /
//! `--output-format`.
//!
//! ```sh
//! online --config examples/bank.toml
//! online --config examples/bank.toml --input other.parquet --resume state.msgpack
//! online --config examples/bank.toml --input today.parquet --resume state.msgpack --predict
//! online --config examples/bank.toml --input ticks.csv --output scored.ndjson
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use online_polars::{Format, RunConfig, run_config};

/// A `--input-format` / `--output-format` value: one of `Format::ALL` by name.
fn parse_format(s: &str) -> Result<Format, String> {
    Format::ALL
        .into_iter()
        .find(|f| f.name() == s)
        .ok_or_else(|| {
            let names: Vec<&str> = Format::ALL.iter().map(|f| f.name()).collect();
            format!("`{s}` is not a format; one of {}", names.join(", "))
        })
}

#[derive(Parser, Debug)]
#[command(name = "online", version, about, long_about = None)]
struct Cli {
    /// TOML file describing the model bank specs and the input/output paths.
    #[arg(long)]
    config: PathBuf,

    /// Override the config's `input`.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Override the config's `output`.
    #[arg(long, conflicts_with = "no_output")]
    output: Option<PathBuf>,

    /// Write no per-row output: the run's product is the state it saves, or
    /// the closed groups it writes (docs/ENHANCEMENTS.md E50). Needs
    /// `save_state` or `closed_groups`, in the config or as a flag. An
    /// accumulator-only spec emits `weight_sum` a row and nothing else, which over
    /// a billion rows is 8 GB of file written so it can be deleted.
    #[arg(long)]
    no_output: bool,

    /// How to read the input (parquet, ipc, csv, ndjson); its extension
    /// decides when unset. Overrides the config's `input_format`.
    #[arg(long, value_parser = parse_format)]
    input_format: Option<Format>,

    /// How to write the output (parquet, ipc, csv, ndjson); its extension
    /// decides when unset. Overrides the config's `output_format`.
    #[arg(long, value_parser = parse_format)]
    output_format: Option<Format>,

    /// Override the config's `chunk_rows`.
    #[arg(long)]
    chunk_rows: Option<usize>,

    /// Resume from this state file (overrides the config's `load_state`).
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Save the final state here (overrides the config's `save_state`).
    #[arg(long)]
    save_state: Option<PathBuf>,

    /// Write the groups that closed during the run to this file, beside the
    /// output (docs/ENHANCEMENTS.md E54). Its format comes from the
    /// extension. Needs a spec with `group_close`; overrides the config's
    /// `closed_groups`.
    #[arg(long)]
    closed_groups: Option<PathBuf>,

    /// Score instead of learn: every row gets the loaded bank's prediction
    /// as it stands and the bank is not updated (sets the config's
    /// `predict`). Needs `--resume` or `load_state`.
    #[arg(long)]
    predict: bool,

    /// Validate the config, open the input and the bank as the run would,
    /// run the bank on no rows of the input's schema, and print the output
    /// schema without reading a row.
    #[arg(long)]
    dry_run: bool,

    /// Suppress per-chunk progress.
    #[arg(long, short)]
    quiet: bool,
}

/// The spec and the key of its model a TOML type error is about. serde
/// reads a model, an internally tagged enum, into a buffer before the
/// variant, so the error points at the `[specs.model]` table and names no
/// key (review 2026-10-06, PC11). Each model's keys are read again alone
/// beside its `type`, and the first that gives the same message is the one.
fn model_key(text: &str, msg: &str) -> Option<String> {
    let doc: toml::Table = toml::from_str(text).ok()?;
    for spec in doc.get("specs")?.as_array()? {
        let Some(model) = spec.get("model").and_then(toml::Value::as_table) else {
            continue;
        };
        let Some(tag) = model.get("type") else {
            continue;
        };
        for (key, value) in model.iter().filter(|(k, _)| k.as_str() != "type") {
            let mut alone = toml::Table::new();
            alone.insert("type".into(), tag.clone());
            alone.insert(key.clone(), value.clone());
            let refused = toml::Value::Table(alone)
                .try_into::<online_polars::ModelKind>()
                .is_err_and(|e| e.message() == msg);
            if refused {
                let name = spec.get("name").and_then(toml::Value::as_str).unwrap_or("");
                return Some(format!("spec {name:?}: model.{key}"));
            }
        }
    }
    None
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("online: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let text = std::fs::read_to_string(&cli.config)
        .map_err(|e| format!("reading {}: {e}", cli.config.display()))?;
    let mut cfg: RunConfig = toml::from_str(&text).map_err(|e| {
        // A Windows path in a TOML basic string is the most common way this
        // fails, and TOML's own message ("too few unicode value digits", from
        // reading `\U` in `C:\Users\...` as an escape) gives no hint why.
        let backslash_hint = if text.contains('\\') {
            "\n\nhint: a backslash starts an escape sequence in a TOML basic string, so a \
             Windows path needs one of:\n  \
             input = 'C:\\data\\in.parquet'     # literal string (single quotes), no escaping\n  \
             input = \"C:\\\\data\\\\in.parquet\"   # basic string, backslashes doubled\n  \
             input = \"C:/data/in.parquet\"      # forward slashes work on Windows too"
        } else {
            ""
        };
        let key = model_key(&text, e.message()).map_or(String::new(), |at| format!("\nat {at}"));
        format!(
            "parsing {}: {}{key}{backslash_hint}",
            cli.config.display(),
            online_polars::name_renamed(&e.to_string())
        )
    })?;

    if let Some(p) = cli.input {
        cfg.input = p;
    }
    if let Some(p) = cli.output {
        cfg.output = p;
    }
    if cli.no_output {
        cfg.output = PathBuf::new();
    }
    if let Some(f) = cli.input_format {
        cfg.input_format = Some(f);
    }
    if let Some(f) = cli.output_format {
        cfg.output_format = Some(f);
    }
    if let Some(n) = cli.chunk_rows {
        cfg.chunk_rows = n;
    }
    if let Some(p) = cli.resume {
        cfg.load_state = Some(p);
    }
    if cli.predict {
        cfg.predict = true;
        // A scoring run closes no group, so the learning run's sidecar is
        // dropped with its `save_state` (review R2, P7).
        cfg.closed_groups = None;
        // One TOML serves both the learning run and the scoring run, and its
        // `save_state` belongs to the former; `--predict` drops it. An
        // explicit `--save-state` is kept below, and `validate` refuses the
        // pair.
        cfg.save_state = None;
    }
    if let Some(p) = cli.save_state {
        cfg.save_state = Some(p);
    }
    if let Some(p) = cli.closed_groups {
        cfg.closed_groups = Some(p);
    }
    // What a spec may leave out, filled before anything reads it (E53).
    cfg.fill_defaults();
    cfg.validate()?;
    // `validate` leaves the input to the run; a dry run wants to know now.
    let input_format = cfg.input_format()?;
    let output_format = if cfg.no_output() {
        None
    } else {
        Some(cfg.output_format()?)
    };

    if cli.dry_run {
        // What the run does first, done now on no rows: the input's schema
        // (the dry run said "config OK" for an input that is not there, task
        // 160, YB12), the bank as the run opens it, and that bank run on a
        // frame of no rows of the schema -- it said "config OK" for a
        // `--resume` state that is not there or holds other specs, and for
        // an input without a column a spec reads (review round 4, SF2).
        cfg.dry_run().map_err(|e| e.to_string())?;
        println!("config OK: {} spec(s)", cfg.specs.len());
        for spec in &cfg.specs {
            println!("  {} ({}):", spec.name, spec.model.kind_name());
            for f in online_polars::output_fields(spec) {
                println!("    {}.{f}", spec.name);
            }
        }
        println!("input:  {} ({})", cfg.input.display(), input_format.name());
        match output_format {
            Some(f) => println!("output: {} ({})", cfg.output.display(), f.name()),
            // `validate` refuses a run with neither, so the list is never
            // empty. The closed groups were left out of it until task 154,
            // and a run whose product they were said it had none.
            None => {
                let products: Vec<String> = [
                    cfg.save_state
                        .as_ref()
                        .map(|p| format!("the state, {}", p.display())),
                    cfg.closed_groups
                        .as_ref()
                        .map(|p| format!("the closed groups, {}", p.display())),
                ]
                .into_iter()
                .flatten()
                .collect();
                println!(
                    "output: none (--no-output); the run's product is {}",
                    products.join(", and ")
                );
            }
        }
        if let Some((p, f)) = cfg.closed_groups_target()? {
            println!("closed groups: {} ({})", p.display(), f.name());
        }
        println!("chunk_rows: {}", cfg.chunk_rows);
        if cfg.predict {
            println!("mode: predict (score against the loaded state, learn nothing)");
        }
        return Ok(());
    }

    let quiet = cli.quiet;
    let stats = run_config(&cfg, |s| {
        if !quiet {
            eprint!("\r{} rows in {} chunks", s.rows, s.chunks);
        }
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    if !quiet {
        eprintln!();
    }
    if cfg.no_output() {
        // Under `--no-output` the path is empty, so name the outcome rather
        // than print a trailing "to " with nothing after it (review
        // 2026-09-18, minor).
        println!(
            "processed {} rows ({} chunks), no output",
            stats.rows, stats.chunks
        );
    } else {
        println!(
            "wrote {} rows ({} chunks) to {}",
            stats.rows,
            stats.chunks,
            cfg.output.display()
        );
    }
    if let Some(p) = &cfg.closed_groups {
        println!("wrote closed groups to {}", p.display());
    }
    if let Some(p) = &cfg.save_state {
        println!("saved state to {}", p.display());
    }
    Ok(())
}
