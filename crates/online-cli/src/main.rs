//! `online` — run a model bank over a parquet stream, config from TOML
//! (docs/PLAN.md §11 task 15).
//!
//! ```sh
//! online --config examples/bank.toml
//! online --config examples/bank.toml --input other.parquet --resume state.msgpack
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use online_polars::{RunConfig, run_config};

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
    #[arg(long)]
    output: Option<PathBuf>,

    /// Override the config's `chunk_rows`.
    #[arg(long)]
    chunk_rows: Option<usize>,

    /// Resume from this state file (overrides the config's `load_state`).
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Save the final state here (overrides the config's `save_state`).
    #[arg(long)]
    save_state: Option<PathBuf>,

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
    let mut cfg: RunConfig =
        toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", cli.config.display()))?;

    if let Some(p) = cli.input {
        cfg.input = p;
    }
    if let Some(p) = cli.output {
        cfg.output = p;
    }
    if let Some(n) = cli.chunk_rows {
        cfg.chunk_rows = n;
    }
    if let Some(p) = cli.resume {
        cfg.load_state = Some(p);
    }
    if let Some(p) = cli.save_state {
        cfg.save_state = Some(p);
    }
    cfg.validate()?;

    if cli.dry_run {
        println!("config OK: {} spec(s)", cfg.specs.len());
        for spec in &cfg.specs {
            println!("  {} ({}):", spec.name, spec.model.kind_name());
            for f in online_polars::output_fields(spec) {
                println!("    {}.{f}", spec.name);
            }
        }
        println!("input:  {}", cfg.input.display());
        println!("output: {}", cfg.output.display());
        println!("chunk_rows: {}", cfg.chunk_rows);
        return Ok(());
    }

    let quiet = cli.quiet;
    let stats = run_config(&cfg, |s| {
        if !quiet {
            eprint!("\r{} rows in {} chunks", s.rows, s.chunks);
        }
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
