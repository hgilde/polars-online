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
use online_polars::{Format, OutputLayout, RunConfig, RunStats, run_config_reported};
use serde::Deserialize;
use toml::Spanned;
use toml::de::{DeTable, DeValue};

/// The binary's allocator, as py-polars' own: jemalloc off Windows, and
/// mimalloc on it, where py-polars uses mimalloc too. On the system
/// allocator the bank ran 1.3x slower a row and 1.8x slower a
/// one-row task than in the wheel: each chunk's output columns and each
/// group's index vectors were handed back to the kernel when freed and
/// faulted in again for the next chunk (review 9, M-5). The wheel keeps
/// `PolarsAllocator`; this is the binary's alone.
#[cfg(not(target_os = "windows"))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc's options, which it reads from this symbol before `main` runs:
/// one arena for every thread, where its default is four a core. Each arena
/// keeps the pages freed in it for ten seconds, for reuse by the threads on
/// that arena alone, and returns them only when one of those threads
/// allocates again; on 56 arenas (14 cores) the reader's, the bank's and
/// the writer's buffers stranded pages in each, and the peak grew with the
/// rows: 0.94 / 1.36 / 1.48 GB at 3M / 12M / 24M rows of the README's
/// memory file, against 0.75 / 0.85 / 0.86 on one arena, at the same speed
/// (a contended arena would show as CPU time: 24.5 s either way at 12M rows
/// on 14 cores; task 270 (o)). `dirty_decay_ms:0` bounded it too, at 18-19%
/// more CPU. `_RJEM_MALLOC_CONF` is read after this and overrides it. The
/// test below holds the option.
#[cfg(not(target_os = "windows"))]
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: Option<&std::ffi::c_char> = Some(
    // SAFETY: the pointer is a `'static` C string literal's first byte,
    // which is what jemalloc's `const char *malloc_conf` points at.
    unsafe { &*c"narenas:1".as_ptr() },
);
#[cfg(target_os = "windows")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

/// A `--output-layout` value: one of `OutputLayout::ALL` by name.
fn parse_layout(s: &str) -> Result<OutputLayout, String> {
    OutputLayout::ALL
        .into_iter()
        .find(|l| l.name() == s)
        .ok_or_else(|| {
            let names: Vec<&str> = OutputLayout::ALL.iter().map(|l| l.name()).collect();
            format!("`{s}` is not an output layout; one of {}", names.join(", "))
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

    /// How the output lays out each spec's struct column: `struct` (the
    /// default) or `flat`, each field a `<spec>.<field>` column, as a CSV
    /// always is. Flat writes a wide spec to parquet several times faster.
    /// Overrides the config's `output_layout`.
    #[arg(long, value_parser = parse_layout)]
    output_layout: Option<OutputLayout>,

    /// Rows per chunk: Polars' `chunk_size`. Overrides the config's
    /// `chunk_size`.
    #[arg(long)]
    chunk_size: Option<usize>,

    /// Threads the bank runs on: a pool of N. Overrides the config's
    /// `threads`; with neither, the process default
    /// (`POLARS_ONLINE_MAX_THREADS`, else one thread per core). The output is
    /// the same at any count.
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

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
fn model_key(doc: &toml::Table, msg: &str) -> Option<String> {
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

/// A parsed TOML model table, as the model-level forwarding reads and
/// rewrites it (`online_polars::ModelTable`). A renamed key keeps the old
/// one's place in the file and a replaced word the old word's, so an error
/// about either names the file's own line.
struct TomlModel<'a, 'i>(&'a mut DeTable<'i>);

impl online_polars::ModelTable for TomlModel<'_, '_> {
    fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    fn word(&self, key: &str) -> Option<String> {
        match self.0.get(key).map(Spanned::get_ref) {
            Some(DeValue::String(s)) => Some(s.to_string()),
            _ => None,
        }
    }
    fn flag(&self, key: &str) -> Option<bool> {
        match self.0.get(key).map(Spanned::get_ref) {
            Some(DeValue::Boolean(b)) => Some(*b),
            _ => None,
        }
    }
    fn shown(&self, key: &str) -> String {
        match self.0.get(key).map(Spanned::get_ref) {
            Some(DeValue::String(s)) => format!("{s:?}"),
            Some(DeValue::Boolean(b)) => b.to_string(),
            Some(DeValue::Integer(i)) => i.to_string(),
            Some(DeValue::Float(f)) => f.to_string(),
            Some(DeValue::Datetime(d)) => d.to_string(),
            Some(DeValue::Array(_)) => "an array".into(),
            Some(DeValue::Table(_)) => "a table".into(),
            None => String::new(),
        }
    }
    fn rename(&mut self, old: &str, new: &str) {
        if let Some((key, value)) = self.0.remove_entry(old) {
            self.0
                .insert(Spanned::new(key.span(), new.to_string().into()), value);
        }
    }
    fn set_word(&mut self, key: &str, word: &str) {
        if let Some(value) = self.0.get_mut(key) {
            *value = Spanned::new(value.span(), DeValue::String(word.to_string().into()));
        }
    }
    fn or_float(&mut self, key: &str, text: &'static str, beside: &str) {
        if self.0.contains_key(key) {
            return;
        }
        // A float TOML parses from its text: `DeFloat` has no other
        // constructor.
        let span = self.0.get_key_value(beside).map_or(0..0, |(k, _)| k.span());
        if let Ok(number) = DeValue::parse(text) {
            self.0.insert(
                Spanned::new(span.clone(), key.to_string().into()),
                Spanned::new(span, number.into_inner()),
            );
        }
    }
}

/// `root`, a parsed TOML config, with each key `table` deprecates read as
/// its new name at any depth, and each model table's renamed keys and words
/// read as the new ones (`online_polars::DEPRECATED`,
/// `online_polars::forward_model`), and a notice for each.
///
/// Done over TOML's own values: it went through JSON, which has no
/// infinity, so a `half_life = inf` anywhere became null and the config was
/// refused; and the text it rewrote moved the lines a later error named
/// (task 269, D-2 and D-7).
fn forward_deprecated_toml(
    root: &mut DeTable<'_>,
    table: &[(&str, &str)],
) -> Result<Vec<String>, String> {
    fn keys(
        t: &mut DeTable<'_>,
        table: &[(&str, &str)],
        notices: &mut Vec<String>,
    ) -> Result<(), String> {
        for (old, new) in table {
            if let Some((key, value)) = t.remove_entry(*old) {
                if t.contains_key(*new) {
                    return Err(format!(
                        "{old} was renamed {new}, and both are given: give {new} alone"
                    ));
                }
                t.insert(Spanned::new(key.span(), (*new).to_string().into()), value);
                notices.push(online_polars::deprecation_notice(old, new));
            }
        }
        for (_, value) in t.iter_mut() {
            walk(value.get_mut(), &mut |t| keys(t, table, notices))?;
        }
        Ok(())
    }
    fn models(t: &mut DeTable<'_>, notices: &mut Vec<String>) -> Result<(), String> {
        let kind = match t.get("type").map(Spanned::get_ref) {
            Some(DeValue::String(s)) => Some(s.to_string()),
            _ => None,
        };
        if let Some(kind) = kind {
            online_polars::forward_model(&kind, &mut TomlModel(t), notices)?;
        }
        for (_, value) in t.iter_mut() {
            walk(value.get_mut(), &mut |t| models(t, notices))?;
        }
        Ok(())
    }
    /// `on` for each table in `v`: `v` itself, or each item of an array.
    fn walk(
        v: &mut DeValue<'_>,
        on: &mut dyn FnMut(&mut DeTable<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        match v {
            DeValue::Table(t) => on(t),
            DeValue::Array(items) => items.iter_mut().try_for_each(|i| walk(i.get_mut(), on)),
            _ => Ok(()),
        }
    }
    let mut notices = Vec::new();
    keys(root, table, &mut notices)?;
    // A key or a word renamed in one model's table, which `table` cannot
    // say: `kmeans`' `k` is `n_clusters` and `hmm`'s `n_components`.
    models(root, &mut notices)?;
    Ok(notices)
}

/// What [`read_config`] gives back: the config or TOML's refusal of it,
/// with the forwarded document as a table, for [`model_key`].
type Read = (Result<RunConfig, toml::de::Error>, Option<toml::Table>);

/// `text`, a TOML config, read as a [`RunConfig`] after its deprecated
/// names are forwarded ([`forward_deprecated_toml`]), and the notices.
/// The document is parsed once and read from its parsed form, so an error
/// names the file's own line and quotes it. Text that is not TOML is
/// refused by the parse.
fn read_config(text: &str, table: &[(&str, &str)]) -> Result<(Read, Vec<String>), String> {
    let mut root = match DeTable::parse(text) {
        Ok(root) => root,
        Err(e) => return Ok(((Err(e), None), Vec::new())),
    };
    let notices = forward_deprecated_toml(root.get_mut(), table)?;
    let doc = toml::Table::deserialize(toml::de::Deserializer::from(root.clone())).ok();
    let cfg = RunConfig::deserialize(toml::de::Deserializer::from(root)).map_err(|mut e| {
        e.set_input(Some(text));
        e
    });
    Ok(((cfg, doc), notices))
}

/// `cfg`'s relative paths, read from `dir`, the config file's directory, so
/// that a config runs the same from any working directory, as a path in a
/// ruff or Cargo config does (task 244, C-2). An empty path, a run with no
/// output, stays empty; an absolute one stays as it is, and so does a URL,
/// which [`RunConfig::localize`] reads or refuses by the name it was given:
/// it was joined under the directory, and refused as a missing
/// `conf/s3://...` (review 9, H-2).
fn relative_to(cfg: &mut RunConfig, dir: &Path) {
    let from_dir = |p: &mut PathBuf| {
        if !p.as_os_str().is_empty() && p.is_relative() && online_polars::url_scheme(p).is_none() {
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
    let ((cfg, doc), notices) = read_config(&text, online_polars::DEPRECATED)
        .map_err(|e| format!("parsing {}: {e}", cli.config.display()))?;
    for notice in notices {
        eprintln!("polars-online: warning: {notice}");
    }
    let mut cfg: RunConfig = cfg.map_err(|e| {
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
        let key = doc
            .as_ref()
            .and_then(|doc| model_key(doc, e.message()))
            .map_or(String::new(), |at| format!("\nat {at}"));
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
    if let Some(l) = cli.output_layout {
        cfg.output_layout = l;
    }
    if let Some(n) = cli.chunk_size {
        cfg.chunk_size = n;
    }
    if let Some(n) = cli.threads {
        cfg.threads = Some(n);
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
    // A `file://` URL is the path it names, and any other URL is refused
    // naming it, before anything reads a path: an `https://` input went to
    // polars' cloud reader (review 9, H-1).
    cfg.localize()?;
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
            Some(f) => {
                // The layout is named where it is not the default, and a CSV
                // is flat whatever it says.
                let layout = match cfg.output_layout {
                    OutputLayout::Flat if f != Format::Csv => ", flat",
                    _ => "",
                };
                println!("output: {} ({}{layout})", cfg.output.display(), f.name());
            }
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
        if let Some(n) = cfg.threads {
            println!("threads: {n}");
        }
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

    /// The binary's jemalloc runs on one arena, read back from jemalloc
    /// itself, so a symbol it no longer reads (a renamed prefix, an
    /// unprefixed build) fails here rather than bring back a peak that grows
    /// with the rows; and `docs/RUNNER.md` names the option this sets.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn jemalloc_runs_on_one_arena_as_the_docs_say() {
        use std::ffi::{c_char, c_int, c_void};
        // jemalloc's `mallctl`, under the `_rjem_` prefix `tikv-jemalloc-sys`
        // builds it with off Windows: the library the binary already links,
        // declared here rather than taken on as a dependency.
        unsafe extern "C" {
            #[link_name = "_rjem_mallctl"]
            fn mallctl(
                name: *const c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> c_int;
        }
        let mut narenas: std::ffi::c_uint = 0;
        let mut len = std::mem::size_of::<std::ffi::c_uint>();
        // SAFETY: `opt.narenas` is an `unsigned`, and `narenas` / `len` are
        // that size, as `mallctl` requires; nothing is written.
        let rc = unsafe {
            mallctl(
                c"opt.narenas".as_ptr(),
                (&raw mut narenas).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(rc, 0, "mallctl(opt.narenas) failed");
        assert_eq!(
            narenas, 1,
            "the binary's jemalloc options went unread, or `_RJEM_MALLOC_CONF` \
             in this environment overrides them"
        );
        let conf = super::MALLOC_CONF.map(|c| {
            // SAFETY: `MALLOC_CONF` points at a C string literal.
            unsafe { std::ffi::CStr::from_ptr(c) }
        });
        assert_eq!(conf, Some(c"narenas:1"));
        let runner = include_str!("../../../docs/RUNNER.md");
        assert!(
            runner.contains("`narenas:1`"),
            "docs/RUNNER.md names `narenas:1`"
        );
    }

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
    /// side by side are refused; the production table leaves the document
    /// as it is (docs/PLAN.md task 198, D2). A renamed key keeps its place
    /// in the file (task 269, D-7).
    #[test]
    fn a_deprecated_toml_key_is_forwarded_with_a_notice() {
        let table = [("half_lyfe", "half_life"), ("rigde", "ridge")];
        let text = "input = \"in.parquet\"\n\n[[specs]]\nname = \"m\"\ntargets = [\"y\"]\n\
                    features = [\"x\"]\nhalf_lyfe = 10.0\n\n[specs.model]\ntype = \"ew_ridge\"\n\
                    rigde = 0.5\n";
        let forwarded = |text: &str, table: &[(&str, &str)]| {
            let mut root = super::DeTable::parse(text).unwrap();
            let notices = super::forward_deprecated_toml(root.get_mut(), table)?;
            let at = match root.get_ref().get("specs").map(super::Spanned::get_ref) {
                Some(super::DeValue::Array(specs)) => match specs[0].get_ref() {
                    super::DeValue::Table(t) => t.get_key_value("half_life").map(|(k, _)| k.span()),
                    _ => None,
                },
                _ => None,
            };
            let doc = <toml::Table as serde::Deserialize>::deserialize(
                toml::de::Deserializer::from(root),
            )
            .unwrap();
            Ok::<_, String>((doc, notices, at))
        };
        let (doc, notices, at) = forwarded(text, &table).unwrap();
        assert_eq!(
            notices,
            [
                online_polars::deprecation_notice("half_lyfe", "half_life"),
                online_polars::deprecation_notice("rigde", "ridge"),
            ]
        );
        let spec = &doc["specs"].as_array().unwrap()[0];
        assert_eq!(spec["half_life"].as_float(), Some(10.0));
        assert_eq!(spec["model"]["ridge"].as_float(), Some(0.5));
        assert!(spec.get("half_lyfe").is_none());
        let old = text.find("half_lyfe").unwrap();
        assert_eq!(at, Some(old..old + "half_lyfe".len()));
        let both = "half_lyfe = 1.0\nhalf_life = 2.0\n";
        let err = forwarded(both, &table).unwrap_err();
        assert!(
            err.contains("half_lyfe") && err.contains("half_life"),
            "{err}"
        );
        let (same, none, _) = forwarded(text, online_polars::DEPRECATED).unwrap();
        assert!(none.is_empty() && same == toml::from_str::<toml::Table>(text).unwrap());
        // Text that is not TOML is refused by the parse, with nothing said
        // of a rename.
        let ((cfg, doc), none) = super::read_config("= [", &table).unwrap();
        assert!(cfg.is_err() && doc.is_none() && none.is_empty());
    }

    /// `bocpd`'s `emission = "robust"` becomes two keys: the renamed one and
    /// a `robust_beta` of 0.1 where none is given, a float as TOML writes it.
    #[test]
    fn a_robust_emission_gains_a_robust_beta_over_toml() {
        let text = "[model]\ntype = \"bocpd\"\nemission = \"robust\"\n";
        let mut root = super::DeTable::parse(text).unwrap();
        let notices =
            super::forward_deprecated_toml(root.get_mut(), online_polars::DEPRECATED).unwrap();
        assert_eq!(notices.len(), 1, "{notices:?}");
        let doc =
            <toml::Table as serde::Deserialize>::deserialize(toml::de::Deserializer::from(root))
                .unwrap();
        let model = doc["model"].as_table().unwrap();
        assert_eq!(model["covariance_type"].as_str(), Some("diag"));
        assert_eq!(model["robust_beta"].as_float(), Some(0.1));
        assert!(model.get("emission").is_none());
    }
}
