//! `polars-online` — run a model bank over a stream of rows, config from TOML
//! (docs/PLAN.md §11 task 15). Reads and writes parquet, ipc, csv and ndjson,
//! each told from its extension or named with `--input-format` /
//! `--output-format`.
//!
//! ```sh
//! polars-online --config examples/bank.toml
//! polars-online --config examples/bank.toml --input other.parquet --load-state bank.state
//! polars-online --config examples/bank.toml --input today.parquet --load-state bank.state --predict
//! polars-online --config examples/bank.toml --input ticks.csv --output scored.ndjson
//! ```

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::LazyLock;

use clap::{CommandFactory, Parser};
use online_polars::{Format, RunConfig, RunStats, run_config_reported};

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

/// What `--version` prints after the name: the package's version, the
/// state schema this build saves in, and the Rust Polars it carries, which a
/// state file and a bug report need beside the version (task 244, C-11).
/// `-V` prints the version alone.
static LONG_VERSION: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}\nstate schema {}\nRust Polars {}",
        env!("CARGO_PKG_VERSION"),
        online_polars::online_core::SCHEMA_VERSION,
        online_polars::POLARS_VERSION,
    )
});

/// `n` and its noun, singular at one: `1 chunk`, `8 chunks`.
fn counted(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// The progress line, rewritten in place with a carriage return as each
/// chunk is read. Only a terminal renders that as intended, so it is drawn
/// only on a terminal standard error, and not under `--quiet`: a log or a
/// pipe gets none of it (task 244, C-9).
struct Progress {
    draw: bool,
}

impl Progress {
    fn new(quiet: bool, terminal: bool) -> Self {
        Self {
            draw: !quiet && terminal,
        }
    }

    /// The line as it stands after `s`. A failed write to standard error
    /// is not the run's failure, so it is dropped.
    fn update(&self, s: RunStats, out: &mut impl Write) {
        if self.draw {
            let _ = write!(
                out,
                "\r{} in {}",
                counted(s.rows, "row"),
                counted(s.chunks, "chunk")
            );
        }
    }

    /// The line ended, once the run is over; a run that read no chunk drew
    /// none, and ends none.
    fn finish(&self, s: RunStats, out: &mut impl Write) {
        if self.draw && s.chunks > 0 {
            let _ = writeln!(out);
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "polars-online",
    version,
    long_version = LONG_VERSION.as_str(),
    about,
    long_about = None
)]
struct Cli {
    /// TOML file describing the model bank specs and the input/output paths.
    /// A relative path inside it is read from the file's own directory, and
    /// one given as a flag from the working directory.
    #[arg(long)]
    config: PathBuf,

    /// Override the config's `input`.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Override the config's `output`.
    #[arg(long, conflicts_with = "no_output")]
    output: Option<PathBuf>,

    /// Write no per-row output: the run's product is the state it saves, or
    /// the closed groups it writes. Needs `save_state` or `closed_groups`,
    /// in the config or as a flag. A run that only accumulates has no use
    /// for a row of output per row of input.
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

    /// Rows per chunk: Polars' `chunk_size`. Overrides the config's
    /// `chunk_size`.
    #[arg(long)]
    chunk_size: Option<usize>,

    /// The old name of `--chunk-size`, refused naming it.
    #[arg(long = "chunk-rows", hide = true, value_name = "N")]
    chunk_rows: Option<String>,

    /// Start from this state file (overrides the config's `load_state`).
    #[arg(long)]
    load_state: Option<PathBuf>,

    /// The old name of `--load-state`, refused naming it.
    #[arg(long, hide = true, value_name = "PATH")]
    resume: Option<String>,

    /// Drop the input's rows the loaded state has learned, so a resume on
    /// input that overlaps it learns each row once, as Python's
    /// `ModelBank.skip_learned` does (sets the config's `skip_learned`).
    /// Needs `--load-state` or `load_state`, and a spec that reads a clock.
    #[arg(long)]
    skip_learned: bool,

    /// Save the final state here (overrides the config's `save_state`).
    #[arg(long)]
    save_state: Option<PathBuf>,

    /// Write the groups that closed during the run to this file, beside the
    /// output. Its format comes from the extension. Needs a spec with `group_close`; overrides the config's
    /// `closed_groups`.
    #[arg(long)]
    closed_groups: Option<PathBuf>,

    /// Score instead of learn: every row gets the loaded bank's prediction
    /// as it stands and the bank is not updated (sets the config's
    /// `predict`). Needs `--load-state` or `load_state`.
    #[arg(long)]
    predict: bool,

    /// Validate the config, open the input and the bank as the run would,
    /// run the bank on no rows of the input's schema, and print the output
    /// schema without reading a row.
    #[arg(long)]
    dry_run: bool,

    /// Suppress the progress line, which is drawn only when standard error
    /// is a terminal.
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

/// The environment variable that turns the unstable label's warning on, as
/// `POLARS_WARN_UNSTABLE` does Polars' own: `1` and nothing else
/// (docs/PLAN.md task 198, D5).
const UNSTABLE_VAR: &str = "POLARS_ONLINE_WARN_UNSTABLE";

/// What the command line says of a formula target under [`UNSTABLE_VAR`]:
/// its tree is read from the TOML file and written into the state, and that
/// written form is not promised.
/// Worded as Python's `UnstableWarning` is. Each line ends in a `\`, which
/// drops the next line's indentation: without them the notice kept six
/// spaces inside each sentence (task 244, C-4).
const FORMULA_UNSTABLE: &str = "warning: a formula target's written form -- its tree in the TOML \
     file and in a saved state -- is considered unstable. It may be changed at any point \
     without it being considered a breaking change (set POLARS_ONLINE_WARN_UNSTABLE=0 or \
     unset it to silence this).";

/// `text`, a TOML config, with each key `table` deprecates read as its new
/// name at any depth, and a notice for each (`online_polars::DEPRECATED`,
/// docs/PLAN.md task 198, D2). The text itself where nothing was renamed, or
/// where it is not TOML at all, which the config's own parse then reports.
fn forward_deprecated_toml(
    text: &str,
    table: &[(&str, &str)],
) -> Result<(String, Vec<String>), String> {
    fn walk(
        v: &mut toml::Value,
        table: &[(&str, &str)],
        notices: &mut Vec<String>,
    ) -> Result<(), String> {
        match v {
            toml::Value::Table(t) => {
                for (old, new) in table {
                    if let Some(value) = t.remove(*old) {
                        if t.contains_key(*new) {
                            return Err(format!(
                                "{old} was renamed {new}, and both are given: give {new} alone"
                            ));
                        }
                        t.insert((*new).to_string(), value);
                        notices.push(online_polars::deprecation_notice(old, new));
                    }
                }
                for (_, value) in t.iter_mut() {
                    walk(value, table, notices)?;
                }
            }
            toml::Value::Array(items) => {
                for value in items {
                    walk(value, table, notices)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let Ok(doc) = toml::from_str::<toml::Table>(text) else {
        return Ok((text.to_string(), Vec::new()));
    };
    let mut doc = toml::Value::Table(doc);
    let mut notices = Vec::new();
    walk(&mut doc, table, &mut notices)?;
    if notices.is_empty() {
        return Ok((text.to_string(), notices));
    }
    let text = toml::to_string(&doc).map_err(|e| format!("rewriting the config: {e}"))?;
    Ok((text, notices))
}

/// `cfg`'s relative paths, read from `dir`, the config file's directory, so
/// that a config runs the same from any working directory, as a path in a
/// ruff or Cargo config does (task 244, C-2). An empty path, a run with no
/// output, stays empty; an absolute one stays as it is.
fn relative_to(cfg: &mut RunConfig, dir: &Path) {
    let from_dir = |p: &mut PathBuf| {
        if !p.as_os_str().is_empty() && p.is_relative() {
            *p = dir.join(&*p);
        }
    };
    from_dir(&mut cfg.input);
    from_dir(&mut cfg.output);
    [
        &mut cfg.load_state,
        &mut cfg.save_state,
        &mut cfg.closed_groups,
    ]
    .into_iter()
    .flatten()
    .for_each(from_dir);
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("polars-online: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A flag task 196 renamed, refused as clap refuses any command line it
/// cannot use (exit status 2), naming the new one: the old name is hidden,
/// and read only to say so.
fn refuse_renamed_flags(cli: &Cli) {
    for (given, old, new) in [
        (cli.chunk_rows.is_some(), "--chunk-rows", "--chunk-size"),
        (cli.resume.is_some(), "--resume", "--load-state"),
    ] {
        if given {
            Cli::command()
                .error(
                    clap::error::ErrorKind::UnknownArgument,
                    format!("{old} was renamed {new}"),
                )
                .exit();
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    refuse_renamed_flags(&cli);
    let text = std::fs::read_to_string(&cli.config)
        .map_err(|e| format!("reading {}: {e}", cli.config.display()))?;
    // A key renamed after 1.0 is read as its new name, with a warning.
    let (text, notices) = forward_deprecated_toml(&text, online_polars::DEPRECATED)
        .map_err(|e| format!("parsing {}: {e}", cli.config.display()))?;
    for notice in notices {
        eprintln!("polars-online: warning: {notice}");
    }
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
            online_polars::name_renamed_run_key(&online_polars::name_renamed(&e.to_string()))
        )
    })?;

    // Before the flags: a path given as a flag is the working directory's.
    relative_to(&mut cfg, cli.config.parent().unwrap_or(Path::new("")));
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
    if let Some(n) = cli.chunk_size {
        cfg.chunk_size = n;
    }
    if let Some(p) = cli.load_state {
        cfg.load_state = Some(p);
    }
    if cli.skip_learned {
        cfg.skip_learned = true;
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
    if std::env::var(UNSTABLE_VAR).is_ok_and(|v| v == "1")
        && cfg.specs.iter().any(|s| s.targets.any_formula())
    {
        eprintln!("polars-online: {FORMULA_UNSTABLE}");
    }
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
        // `--load-state` state that is not there or holds other specs, and for
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
        println!("chunk_size: {}", cfg.chunk_size);
        if cfg.predict {
            println!("mode: predict (score against the loaded state, learn nothing)");
        }
        return Ok(());
    }

    let progress = Progress::new(cli.quiet, std::io::stderr().is_terminal());
    let (stats, readiness) = run_config_reported(&cfg, |s| {
        progress.update(s, &mut std::io::stderr());
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    progress.finish(stats, &mut std::io::stderr());
    let (rows, chunks) = (counted(stats.rows, "row"), counted(stats.chunks, "chunk"));
    if cfg.no_output() {
        // Under `--no-output` the path is empty, so name the outcome rather
        // than print a trailing "to " with nothing after it (review
        // 2026-09-18, minor).
        println!("processed {rows} ({chunks}), no output");
    } else {
        println!("wrote {rows} ({chunks}) to {}", cfg.output.display());
    }
    // Where each spec's groups stand on readiness after their last row
    // (docs/WARMUP-AND-CONVERGENCE.md §7.9): the groups whose last row was
    // withheld, by reason, and those with a coefficient more ridge than
    // data; nothing for a spec with neither.
    for line in readiness {
        println!("{line}");
    }
    if let Some(p) = &cfg.closed_groups {
        println!("wrote closed groups to {}", p.display());
    }
    if let Some(p) = &cfg.save_state {
        println!("saved state to {}", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use online_polars::RunStats;

    /// A relative path is read from the config's directory, an absolute one
    /// stays as it is, and an empty one -- `output` left out, a run with no
    /// output -- stays empty, rather than become the directory itself.
    #[test]
    fn a_relative_path_is_the_configs_and_an_empty_one_stays_empty() {
        let abs = std::env::temp_dir().join("c.parquet");
        let text = format!(
            "input = 'in.parquet'\nsave_state = 's.state'\nclosed_groups = '{}'\n\
             [[specs]]\nname = 'm'\ntargets = ['y']\nfeatures = ['x']\nhalf_life = 1.0\n\
             [specs.model]\ntype = 'ewridge'\n",
            abs.display()
        );
        let mut cfg: online_polars::RunConfig = toml::from_str(&text).unwrap();
        let dir = std::path::Path::new("conf");
        super::relative_to(&mut cfg, dir);
        assert_eq!(cfg.input, dir.join("in.parquet"));
        assert_eq!(cfg.save_state, Some(dir.join("s.state")));
        assert_eq!(cfg.closed_groups, Some(abs));
        assert!(cfg.output.as_os_str().is_empty() && cfg.load_state.is_none());
    }

    /// The progress line is drawn on a terminal without `--quiet`, and
    /// nowhere else; it counts in the singular at one, and a run that read
    /// no chunk ends no line.
    #[test]
    fn the_progress_line_is_drawn_on_a_terminal_without_quiet_alone() {
        for (quiet, terminal, drawn) in [
            (false, true, true),
            (true, true, false),
            (false, false, false),
            (true, false, false),
        ] {
            let progress = super::Progress::new(quiet, terminal);
            let mut out = Vec::new();
            progress.update(RunStats { rows: 7, chunks: 1 }, &mut out);
            let end = RunStats {
                rows: 50,
                chunks: 8,
            };
            progress.update(end, &mut out);
            progress.finish(end, &mut out);
            let want = if drawn {
                "\r7 rows in 1 chunk\r50 rows in 8 chunks\n"
            } else {
                ""
            };
            assert_eq!(String::from_utf8(out).unwrap(), want, "{quiet} {terminal}");
        }
        let mut out = Vec::new();
        super::Progress::new(false, true).finish(RunStats::default(), &mut out);
        assert!(out.is_empty());
    }
    /// A key the table deprecates is read as its new name at any depth, a
    /// spec's own and its model's, with a notice each; an old and a new name
    /// side by side are refused; the production table, empty before 1.0,
    /// leaves the text as it is (docs/PLAN.md task 198, D2).
    #[test]
    fn a_deprecated_toml_key_is_forwarded_with_a_notice() {
        let table = [("half_lyfe", "half_life"), ("rigde", "ridge")];
        let text = "input = \"in.parquet\"\n\n[[specs]]\nname = \"m\"\ntargets = [\"y\"]\n\
                    features = [\"x\"]\nhalf_lyfe = 10.0\n\n[specs.model]\ntype = \"ew_ridge\"\n\
                    rigde = 0.5\n";
        let (out, notices) = super::forward_deprecated_toml(text, &table).unwrap();
        assert_eq!(
            notices,
            [
                online_polars::deprecation_notice("half_lyfe", "half_life"),
                online_polars::deprecation_notice("rigde", "ridge"),
            ]
        );
        let doc: toml::Table = toml::from_str(&out).unwrap();
        let spec = &doc["specs"].as_array().unwrap()[0];
        assert_eq!(spec["half_life"].as_float(), Some(10.0));
        assert_eq!(spec["model"]["ridge"].as_float(), Some(0.5));
        assert!(spec.get("half_lyfe").is_none());
        let both = "half_lyfe = 1.0\nhalf_life = 2.0\n";
        let err = super::forward_deprecated_toml(both, &table).unwrap_err();
        assert!(
            err.contains("half_lyfe") && err.contains("half_life"),
            "{err}"
        );
        let (same, none) = super::forward_deprecated_toml(text, online_polars::DEPRECATED).unwrap();
        assert!(none.is_empty() && same == text);
        // Text that is not TOML is left for the config's own parse to report.
        let (bad, none) = super::forward_deprecated_toml("= [", &table).unwrap();
        assert!(none.is_empty() && bad == "= [");
    }
}
