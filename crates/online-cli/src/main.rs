//! `online` — run a model bank over a parquet stream, config from TOML.
//!
//! See `docs/PLAN.md` §11 task 15. Scaffold only: this currently just parses arguments.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "online", version, about, long_about = None)]
struct Cli {
    /// TOML file describing the model bank specs and the input/output paths.
    #[arg(long)]
    config: std::path::PathBuf,
}

fn main() {
    let cli = Cli::parse();
    eprintln!(
        "online-cli scaffold: would run the bank from {}",
        cli.config.display()
    );
}
