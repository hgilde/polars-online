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

    /// Write no per-row output: the run's product is the state it saves
    /// (docs/ENHANCEMENTS.md E50). Needs `save_state`, in the config or with
    /// `--save-state`. An accumulator-only spec emits `n_eff` a row and
    /// nothing else, which over a billion rows is 8 GB of file written so it
    /// can be deleted.
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

    /// Score instead of learn: every row gets the loaded bank's prediction
    /// as it stands and the bank is not updated (sets the config's
    /// `predict`). Needs `--resume` or `load_state`.
    #[arg(long)]
    predict: bool,

    /// Validate the config and print the output schema without running.
    #[arg(long)]
    dry_run: bool,

    /// Suppress per-chunk progress.
    #[arg(long, short)]
    quiet: bool,
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
        format!("parsing {}: {e}{backslash_hint}", cli.config.display())
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
        // One TOML serves both the learning run and the scoring run, and its
        // `save_state` belongs to the former; `--predict` drops it. An
        // explicit `--save-state` is kept below, and `validate` refuses the
        // pair.
        cfg.save_state = None;
    }
    if let Some(p) = cli.save_state {
        cfg.save_state = Some(p);
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
            None => println!(
                "output: none (--no-output); the run's product is {}",
                cfg.save_state
                    .as_ref()
                    .map_or_else(|| "nothing".into(), |p| p.display().to_string())
            ),
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
    println!(
        "wrote {} rows ({} chunks) to {}",
        stats.rows,
        stats.chunks,
        cfg.output.display()
    );
    if let Some(p) = &cfg.save_state {
        println!("saved state to {}", p.display());
    }
    Ok(())
}
