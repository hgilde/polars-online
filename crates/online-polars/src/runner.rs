//! Streaming runner behind the `online` CLI (docs/PLAN.md
//! §11 task 15; docs/ENHANCEMENTS.md E32).
//!
//! Any source polars can scan comes in, any file format polars can write goes
//! out, and the numbers in between are the bank's. A run is a three-stage
//! pipeline:
//!
//! 1. a reader thread hands over frames in stream order: polars' streaming
//!    engine reading a plan in `chunk_size` rows (`sink_batches`), or an
//!    iterator of frames the caller already has ([`Input::Batches`], for a
//!    Rust caller; the Python runner that fed it was removed in task 83);
//! 2. this thread feeds each frame to the [`Bank`] and appends the outputs;
//! 3. a writer thread encodes and writes the augmented frame in the output
//!    format, through a temporary that is renamed into place at the end.
//!
//! Each stage holds one frame and passes the next over a channel of capacity
//! one, so memory stays O(chunk) rather than O(data), and the read, the fit
//! and the write overlap. Chunking never changes the numbers (docs/PLAN.md §9
//! class 2); it only trades memory for overhead.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use polars::prelude::*;
use polars_utils::pl_path::PlRefPath;
use serde::{Deserialize, Serialize};

use crate::atomic::AtomicFile;
use crate::bank::{Bank, Learned};
use crate::spec::Spec;

/// Writing the output through a temporary is filesystem work, and polars'
/// error type is what the runner returns: its `IO` variant, kind intact, so
/// a caller can tell a file that could not be written or read from a run
/// that was refused. `what` and `path` say which file: the `io::Error`
/// alone does not.
fn io_err(what: &str, path: &Path, e: std::io::Error) -> PolarsError {
    let msg = format!("{what} {}: {e}", path.display());
    PolarsError::IO {
        error: Arc::new(e),
        msg: Some(msg.into()),
    }
}

/// Whether `path` can be written once the run is done, or the error naming
/// it: a directory that is not there is reported before a run, not after the
/// stream it would have cost, and so is a path that is a directory itself,
/// which the write at the end cannot replace -- a `--save-state` naming a
/// directory was found after the output had been published (task 160,
/// YB7). `what` is as for [`io_err`].
fn check_parent(what: &str, path: &Path) -> PolarsResult<()> {
    if path.is_dir() {
        let e = std::io::Error::new(
            std::io::ErrorKind::IsADirectory,
            format!("{} is a directory", path.display()),
        );
        return Err(io_err(what, path, e));
    }
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if parent.is_dir() {
        return Ok(());
    }
    let e = std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{} is not a directory", parent.display()),
    );
    Err(io_err(what, path, e))
}

/// A file format the runner reads and writes.
///
/// Reading goes through polars' lazy scans, so a source is read the way
/// `scan_parquet` / `scan_ipc` / `scan_csv` / `scan_ndjson` read it, with
/// their defaults (a CSV's dtypes are inferred from its first rows). Writing
/// carries the bank's struct columns as they are in every format but CSV,
/// which has no nested values: there each spec's struct is flattened into
/// `<spec>.<field>` columns, and a list field (`coef`) is refused with a
/// message naming it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Parquet,
    Ipc,
    Csv,
    Ndjson,
}

impl Format {
    /// Every format, in the order the docs list them.
    pub const ALL: [Format; 4] = [Format::Parquet, Format::Ipc, Format::Csv, Format::Ndjson];

    /// The spec / TOML name: `parquet`, `ipc`, `csv`, `ndjson`.
    pub fn name(self) -> &'static str {
        match self {
            Format::Parquet => "parquet",
            Format::Ipc => "ipc",
            Format::Csv => "csv",
            Format::Ndjson => "ndjson",
        }
    }

    /// The format a path's extension names: `parquet`/`pq`, `ipc`/`arrow`/
    /// `feather`, `csv`, `ndjson`/`jsonl`. Case-insensitive.
    pub fn from_path(path: &Path) -> Result<Format, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "parquet" | "pq" => Ok(Format::Parquet),
            "ipc" | "arrow" | "feather" => Ok(Format::Ipc),
            "csv" => Ok(Format::Csv),
            "ndjson" | "jsonl" => Ok(Format::Ndjson),
            _ => Err(format!(
                "cannot tell the format of `{}` from its extension (parquet, pq, ipc, arrow, \
                 feather, csv, ndjson, jsonl); name it with input_format / output_format",
                path.display()
            )),
        }
    }

    /// A lazy scan of `path` in this format, with polars' defaults.
    pub fn scan(self, path: &Path) -> PolarsResult<LazyFrame> {
        let path = PlRefPath::try_from_pathbuf(path.to_path_buf())?;
        match self {
            Format::Parquet => LazyFrame::scan_parquet(path, ScanArgsParquet::default()),
            Format::Ipc => {
                LazyFrame::scan_ipc(path, IpcScanOptions::default(), UnifiedScanArgs::default())
            }
            Format::Csv => LazyCsvReader::new(path).finish(),
            Format::Ndjson => LazyJsonLineReader::new(path).finish(),
        }
    }
}

/// A run description, deserialized from TOML by the CLI (and, until task 83
/// removed it, from JSON by the Python runner). An unknown key is refused,
/// naming the keys there
/// are, so a misspelt one cannot silently fall back to its default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Input path. Empty when the caller supplies the source itself
    /// ([`run_config_on`]), as a Rust caller with a `LazyFrame` of its own
    /// does.
    #[serde(default)]
    pub input: PathBuf,
    /// Output path. Empty for a run whose product is its state
    /// (docs/ENHANCEMENTS.md E50): `save_state` is then required, since a run
    /// that writes nothing and saves nothing has done nothing.
    #[serde(default)]
    pub output: PathBuf,
    /// How to read `input`; its extension decides when unset.
    #[serde(default)]
    pub input_format: Option<Format>,
    /// How to write `output`; its extension decides when unset.
    #[serde(default)]
    pub output_format: Option<Format>,
    /// Rows per chunk, [`DEFAULT_CHUNK_SIZE`] when unset. Chunking never
    /// changes the numbers (docs/PLAN.md §9 class 2); it only trades memory
    /// for overhead.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Load the bank state from here before running (resume).
    #[serde(default)]
    pub load_state: Option<PathBuf>,
    /// Drop the rows of the input the loaded state has learned, so a resume
    /// on input that overlaps it learns each row once: in every spec that
    /// reads a clock, a row is kept when its clock is after its group's last
    /// one, its group is new to the bank, or its clock is null
    /// (`ModelBank.skip_learned` in Python; docs/PLAN.md task 196, N26).
    /// Requires `load_state` and a spec that reads a clock.
    #[serde(default)]
    pub skip_learned: bool,
    /// Save the bank state here after running.
    #[serde(default)]
    pub save_state: Option<PathBuf>,
    /// Columns to keep from the input; all of them when empty.
    #[serde(default)]
    pub keep_columns: Vec<String>,
    /// Score instead of learn: every row gets the prediction the loaded bank
    /// makes for it as it stands, and the bank is not updated
    /// (docs/ENHANCEMENTS.md E31). Requires `load_state`; `save_state` is
    /// refused, since there is nothing new to save.
    #[serde(default)]
    pub predict: bool,
    /// Write the groups that closed during the run here, as a sidecar frame
    /// beside `output` (docs/ENHANCEMENTS.md E54). Its format comes from the
    /// extension, exactly as `output`'s does.
    ///
    /// Needs at least one spec with `group_close`; refused with `predict`,
    /// which closes nothing. `output` may be empty at the same time: that is
    /// the accumulate-only pass whose product is the closed rows and the
    /// state.
    #[serde(default)]
    pub closed_groups: Option<PathBuf>,
    /// The model specs to run.
    pub specs: Vec<Spec>,
}

/// Rows per chunk when a config does not say: enough to amortize the
/// per-chunk work, small enough to keep three frames of it in memory.
pub const DEFAULT_CHUNK_SIZE: usize = 100_000;

fn default_chunk_size() -> usize {
    DEFAULT_CHUNK_SIZE
}

/// The run keys task 196 renamed (docs/PLAN.md §18, N2): `chunk_size` is
/// Polars' name on the call it feeds (`collect_batches(chunk_size=)`). A
/// config naming the old key is refused naming the new one, as a spec's old
/// key is ([`crate::RENAMED`]).
pub const RENAMED_RUN_KEYS: &[(&str, &str)] = &[("chunk_rows", "chunk_size")];

/// `msg`, a config's deserialization error, with the rename named when the
/// key it refuses as unknown is an old run key.
pub fn name_renamed_run_key(msg: &str) -> String {
    for (old, new) in RENAMED_RUN_KEYS {
        if msg.contains(&format!("unknown field `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    msg.to_string()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub rows: usize,
    pub chunks: usize,
}

impl RunConfig {
    /// Everything but the input, which [`run_config`] checks and
    /// [`run_config_on`] does not need.
    /// [`Spec::fill_defaults`] for every spec, so a config parsed from TOML
    /// carries the same specs a Python caller would have built
    /// (docs/ENHANCEMENTS.md E53). Called by the CLI after the config is
    /// read; idempotent.
    pub fn fill_defaults(&mut self) {
        for s in self.specs.iter_mut() {
            s.fill_defaults();
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.specs.is_empty() {
            return Err("config has no [[specs]] entries".into());
        }
        if self.chunk_size == 0 {
            return Err("chunk_size must be > 0".into());
        }
        // Named first: a scoring run learns nothing, so it can save no state
        // and close no group, and the refusal below asked for one of the two
        // (task 160, YB13).
        if self.no_output() && self.predict {
            return Err(
                "predict = true (--predict) scores the rows and learns nothing, so its output is \
                 its only product: it needs an `output` path, and has none (--no-output, or no \
                 `output` in the config)"
                    .into(),
            );
        }
        if self.no_output() && self.save_state.is_none() && self.closed_groups.is_none() {
            return Err(
                "a run needs somewhere to put its work: an `output` path, `closed_groups`, or \
                 `save_state` for a run whose product is the state (--no-output)"
                    .into(),
            );
        }
        if let Some(p) = &self.closed_groups {
            if self.predict {
                return Err(
                    "predict = true does not learn, so no group ever closes and closed_groups \
                     would be an empty file; drop one or the other"
                        .into(),
                );
            }
            if !self.specs.iter().any(|s| s.group_close.is_some()) {
                return Err(
                    "closed_groups names a file but no spec closes groups; add group_close = \
                     \"monotone\" or \"session\" to the spec whose groups should be emitted"
                        .into(),
                );
            }
            Format::from_path(p)?;
        }
        if self.skip_learned && self.load_state.is_none() {
            return Err(
                "skip_learned = true (--skip-learned) needs load_state (--load-state): a fresh \
                 bank has learned nothing to skip"
                    .into(),
            );
        }
        if self.predict {
            if self.load_state.is_none() {
                return Err(
                    "predict = true needs load_state: a fresh bank has nothing to score with"
                        .into(),
                );
            }
            if self.save_state.is_some() {
                return Err(
                    "predict = true does not update the bank, so save_state has nothing to save; \
                     drop one or the other"
                        .into(),
                );
            }
        }
        if !self.no_output() {
            self.output_format()?;
        }
        // Each spec's checks and the bank's -- a duplicate name, a `seqtest`
        // comparison naming a spec the bank has not got or a residual a side
        // does not emit -- so that a dry run reports them and not the first
        // chunk. `Bank::new` fills each spec before it checks it
        // (`Spec::check`); a validation of its own here refused a spec the
        // bank would have filled (review 2026-09-12, S25).
        let bank = Bank::new(self.specs.clone())?;
        // And what the bank refuses at the first chunk of a run that keeps
        // its predictions, a formula target's embargo short of its window:
        // the dry run passed it and the run then refused it (task 154).
        if !self.no_output()
            && !self.predict
            && let Some(why) = bank.fit_predict_refusal()
        {
            return Err(why.to_string());
        }
        Ok(())
    }

    /// Does this run write per-row output at all (docs/ENHANCEMENTS.md E50)?
    ///
    /// An accumulator-only spec emits `weight_sum` a row and nothing else; over a
    /// billion rows that is 8 GB of file written so it can be deleted. When
    /// the product of the run is the state, there is nothing to write.
    pub fn no_output(&self) -> bool {
        self.output.as_os_str().is_empty()
    }

    /// `input_format`, or what the input path's extension says.
    pub fn input_format(&self) -> Result<Format, String> {
        if self.input.as_os_str().is_empty() {
            return Err("input is required: the config's `input`, or --input".into());
        }
        self.input_format
            .map_or_else(|| Format::from_path(&self.input), Ok)
    }

    /// `output_format`, or what the output path's extension says.
    pub fn output_format(&self) -> Result<Format, String> {
        self.output_format
            .map_or_else(|| Format::from_path(&self.output), Ok)
    }

    /// The `closed_groups` sidecar's path and format, when there is one.
    pub fn closed_groups_target(&self) -> Result<Option<(&Path, Format)>, String> {
        match &self.closed_groups {
            Some(p) => Ok(Some((p.as_path(), Format::from_path(p)?))),
            None => Ok(None),
        }
    }

    /// The lazy scan of `input` this config describes, `keep_columns` applied.
    pub fn scan(&self) -> PolarsResult<LazyFrame> {
        let format = self
            .input_format()
            .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        Ok(self.keep(format.scan(&self.input)?))
    }

    /// `keep_columns` as a projection; the frame itself when empty.
    fn keep(&self, lf: LazyFrame) -> LazyFrame {
        if self.keep_columns.is_empty() {
            lf
        } else {
            let cols: Vec<Expr> = self.keep_columns.iter().map(|c| col(c.as_str())).collect();
            lf.select(cols)
        }
    }

    /// What `online --dry-run` checks beyond [`Self::validate`], reading no
    /// row: the input's schema as the run's scan resolves it -- a parquet
    /// footer, a CSV's first rows, a glob's first file, `keep_columns`
    /// applied -- the bank as the run opens it ([`Self::open_bank`]), and
    /// that bank run, as the run would run it, on a frame of no rows of that
    /// schema. So a `load_state` that is not there or holds other specs, and
    /// an input or a `keep_columns` without a column a spec reads, are
    /// refused here as the run refuses them at its first step; a dry run
    /// said "config OK" for all three (review round 4, SF2). The bank is
    /// dropped: nothing is written.
    ///
    /// # Errors
    ///
    /// The scan's error, its message led by `input <path>: `; then
    /// [`Self::open_bank`]'s; then [`Bank::fit_predict`]'s, or
    /// [`Bank::predict`]'s for a scoring run.
    pub fn dry_run(&self) -> PolarsResult<()> {
        let schema = self
            .scan()
            .and_then(|mut lf| lf.collect_schema())
            .map_err(|e| e.wrap_msg(|m| format!("input {}: {m}", self.input.display())))?;
        let mut bank = self.open_bank()?;
        let empty = DataFrame::empty_with_schema(&schema);
        if self.skip_learned {
            // What the run takes before its first row: a bank with no clock
            // to resume by is refused here too.
            let learned = bank
                .learned()
                .map_err(|e| polars_err!(ComputeError: "{}", e))?;
            learned.unlearned(&empty)?;
        }
        augment(&mut bank, empty, self.predict, self.no_output(), 0).map(|_| ())
    }

    /// The bank this config starts from: loaded from `load_state`, or fresh.
    /// A file that cannot be read is an `IO` error; one that is not a bank
    /// this build loads, or whose specs are not the config's, is a
    /// `ComputeError` saying why (`Bank::load_bytes`).
    pub fn open_bank(&self) -> PolarsResult<Bank> {
        match &self.load_state {
            Some(p) => {
                let bytes = std::fs::read(p).map_err(|e| io_err("loading state", p, e))?;
                Bank::load_bytes(&bytes, Some(&self.specs))
                    .map_err(|e| polars_err!(ComputeError: "loading state {}: {}", p.display(), e))
            }
            None => Bank::new(self.specs.clone()).map_err(|e| polars_err!(ComputeError: "{}", e)),
        }
    }
}

/// Where a run's rows come from.
// One per run, so the variant sizes are irrelevant; `Input::Lazy(lf)` reads
// better than a `Box` at every call site.
#[allow(clippy::large_enum_variant)]
pub enum Input<'a> {
    /// A polars plan, read by the streaming engine in chunks of
    /// `chunk_size` rows: a scan, a query, an in-memory frame's `lazy()`.
    Lazy(LazyFrame),
    /// Frames the caller produces, in stream order and in whatever sizes it
    /// has them (`chunk_size` does not re-chunk them). `schema` is the frames'
    /// schema, for the output of a stream that turns out to have none. An
    /// error ends the run with that error. The iterator is pulled on the
    /// reader thread, so it must be `Send`; it is dropped there when the run
    /// ends, early or not.
    Batches {
        frames: Box<dyn Iterator<Item = PolarsResult<DataFrame>> + Send + 'a>,
        schema: Schema,
    },
}

/// Where a run's augmented frames go.
pub enum Output<'a> {
    /// A file in `format`, written through a temporary sibling that is
    /// renamed into place once the run has completed, so a run that fails
    /// leaves any previous output where it was.
    File { path: &'a Path, format: Format },
    /// Each augmented frame, in stream order, on the calling thread. For a
    /// destination polars cannot write to: a database, a socket, a test.
    Batches(&'a mut dyn FnMut(DataFrame) -> PolarsResult<()>),
    /// Nowhere: the run's product is the state it saves, and the per-row
    /// output would be I/O nobody reads (docs/ENHANCEMENTS.md E50). An
    /// accumulator-only spec over a billion rows still emits `weight_sum` a row,
    /// which is 8 GB of file to write and delete.
    Discard,
}

/// The knobs of [`run`] that are not the source or the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOptions {
    /// Rows per chunk. See [`RunConfig::chunk_size`].
    pub chunk_size: usize,
    /// Score instead of learn. See [`RunConfig::predict`].
    pub predict: bool,
    /// The run keeps no prediction (`--no-output`): `ModelBank.fit`'s run,
    /// which takes any embargo under a formula target (review R2, P4).
    pub learn_only: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            predict: false,
            learn_only: false,
        }
    }
}

/// Run a config end to end: scan `input`, run the bank, write `output`, save
/// the state. `progress` is called after each chunk with the running stats,
/// so the CLI can print without this crate knowing about stdout; an error
/// from it ends the run with that error (the output is not published).
///
/// # Errors
///
/// Before a row is read: `ComputeError` for a config [`RunConfig::validate`]
/// refuses or a bank [`Bank::new`] does; `PolarsError::IO` -- carrying the
/// `io::Error` under a message naming the path -- for a `load_state` that
/// cannot be read, and for a `save_state` whose directory is not there or
/// that is a directory itself, checked before the run because finding out
/// after it would leave the output written and the state lost (the output
/// and `closed_groups` paths are held to the same); `ComputeError` as `loading state
/// <path>: ...` for a `load_state` that is not a bank this build loads or
/// whose specs are not the config's. During the run (the scan is lazy):
/// polars' own error for `input` (a missing file is its `IO`, naming the
/// path; `keep_columns` naming a column it has not got is `ColumnNotFound`),
/// `PolarsError::IO` as `writing <output>: ...` for the output, and
/// [`Bank::fit_predict`]'s for the data. Whatever ends the run leaves the
/// previous `output` in place and `save_state` unwritten: the state is saved
/// last, as `PolarsError::IO` `saving state <path>: ...` if that fails, so a
/// state file always has an output to go with it.
pub fn run_config(
    cfg: &RunConfig,
    progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<RunStats> {
    run_config_reported(cfg, progress).map(|(stats, _)| stats)
}

/// [`run_config`], with the run's closing lines on readiness
/// (docs/WARMUP-AND-CONVERGENCE.md §7.9; docs/PLAN.md task 116, H): one per
/// spec with something to say -- the groups whose last row was withheld,
/// counted by `withheld_reason`, and the groups whose smallest data share
/// is below 0.5 -- and none for a spec with neither. What the command line
/// prints after "wrote N rows".
///
/// # Errors
///
/// [`run_config`]'s.
pub fn run_config_reported(
    cfg: &RunConfig,
    progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<(RunStats, Vec<String>)> {
    cfg.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let format = cfg
        .input_format()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    // `keep_columns` is applied once, by `run_config_on`.
    run_config_on_reported(cfg, Input::Lazy(format.scan(&cfg.input)?), progress)
}

/// [`run_config`] over a source of the caller's choosing instead of the
/// config's `input` path: any `LazyFrame` -- a scan of something the config
/// cannot name, a query, an in-memory frame -- or frames the caller already
/// has. `keep_columns` still applies, to a plan as a `select` (so the scan
/// reads only those columns) and to frames one by one.
///
/// # Errors
///
/// [`run_config`]'s, with the source's own in place of the scan's.
pub fn run_config_on(
    cfg: &RunConfig,
    input: Input<'_>,
    progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<RunStats> {
    run_config_on_reported(cfg, input, progress).map(|(stats, _)| stats)
}

/// [`run_config_on`], with [`run_config_reported`]'s closing lines.
///
/// # Errors
///
/// [`run_config_on`]'s.
pub fn run_config_on_reported(
    cfg: &RunConfig,
    input: Input<'_>,
    progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<(RunStats, Vec<String>)> {
    cfg.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    // A run with no output has no format to work out (E50).
    let format = if cfg.no_output() {
        Format::Parquet
    } else {
        cfg.output_format()
            .map_err(|e| polars_err!(ComputeError: "{}", e))?
    };
    let input = match input {
        Input::Lazy(lf) => Input::Lazy(cfg.keep(lf)),
        Input::Batches { frames, schema } if cfg.keep_columns.is_empty() => {
            Input::Batches { frames, schema }
        }
        Input::Batches { frames, schema } => {
            let cols = cfg.keep_columns.clone();
            let schema = DataFrame::empty_with_schema(&schema)
                .select(cols.iter().map(String::as_str))?
                .schema()
                .as_ref()
                .clone();
            let frames =
                frames.map(move |r| r.and_then(|df| df.select(cols.iter().map(String::as_str))));
            Input::Batches {
                frames: Box::new(frames),
                schema,
            }
        }
    };
    if let Some(p) = &cfg.save_state {
        check_parent("saving state", p)?;
    }
    // The output directory too, so an unwritable one is refused before the
    // bank learns a chunk rather than by the writer thread mid-run (review
    // 2026-09-18, minor). Skipped under `--no-output`, where the path is empty.
    if !cfg.no_output() {
        check_parent("writing output", &cfg.output)?;
    }
    let closed_target = cfg
        .closed_groups_target()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    if let Some((p, _)) = closed_target {
        check_parent("writing closed groups", p)?;
    }
    let mut bank = cfg.open_bank()?;
    // What the loaded state has learned, taken once before the first row,
    // so every chunk is filtered against the same positions.
    let learned = if cfg.skip_learned {
        Some(
            bank.learned()
                .map_err(|e| polars_err!(ComputeError: "{}", e))?,
        )
    } else {
        None
    };
    let opts = RunOptions {
        chunk_size: cfg.chunk_size,
        predict: cfg.predict,
        learn_only: cfg.no_output(),
    };
    let out = if cfg.no_output() {
        Output::Discard
    } else {
        Output::File {
            path: &cfg.output,
            format,
        }
    };
    // The sidecar is written as the run goes -- drained after every chunk --
    // and published with the output, before the state, so a state file
    // always has the closed rows that go with it. A run in which nothing
    // closed writes an empty frame with the schema, as an empty output does.
    let mut last = LastReasons::default();
    let stats = run_with(
        &mut bank,
        input,
        out,
        closed_target,
        learned.as_ref(),
        Some(&mut last),
        opts,
        progress,
    )?;
    if let Some(p) = &cfg.save_state {
        bank.save(p).map_err(|e| io_err("saving state", p, e))?;
    }
    Ok((stats, readiness_lines(&bank, &last)))
}

/// The run's error when the writer thread went away mid-run; the writer's
/// own error replaces it on the way out.
const WRITER_STOPPED: &str = "the writer stopped";

/// What the reader hands the run: frames in stream order, then how the
/// query ended.
enum Read {
    Chunk(DataFrame),
    End(PolarsResult<()>),
}

/// What the run hands the writer. A sender dropped without `End` is a run
/// that failed: the writer discards its temporary instead of publishing it.
enum Write_ {
    Chunk(DataFrame),
    End,
}

/// Stream `input` through `bank` and deliver the augmented frames -- the
/// input columns plus one struct column per spec -- to `output`. The bank is
/// left where the stream ended, so the caller can save it or keep feeding
/// it. `progress` is called after each chunk; an error from it ends the run.
///
/// The read and the write each run on a thread of their own, a chunk ahead
/// of and behind the bank; see the module docs.
///
/// # Errors
///
/// The source's, the bank's ([`Bank::fit_predict`] or [`Bank::predict`]),
/// the writer's (`PolarsError::IO` naming the file) or `progress`'s,
/// whichever comes first; the bank is left as it was after the last chunk
/// it accepted -- unless a window passed a refusing `window_budget`, which
/// is found as the rows go in and leaves the bank refusing to go on
/// ([`Bank::fit_predict`]) -- and a file output is not published. A
/// `closed_groups` sidecar is: the bank's closed groups are drained into it
/// as the run goes, so it is published with whatever it drained, complete
/// or not, and the bank's queue holds none of them.
pub fn run(
    bank: &mut Bank,
    input: Input<'_>,
    output: Output<'_>,
    opts: RunOptions,
    progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<RunStats> {
    run_with(bank, input, output, None, None, None, opts, progress)
}

/// [`run`], with the bank's closed groups (E54) drained after every chunk
/// into `closed` -- a file, in its format -- written through the output's
/// atomic path and published by a run that reached its end, and by one that
/// failed after draining anything (the second review of 2026-09-15, F2). The
/// bank's queue is one chunk deep, where a drain after the last chunk held
/// every row that closed for the length of the run (review 2026-09-12, P5).
/// With `skip`, each chunk keeps only the rows the loaded state has not
/// learned before the bank sees it (`--skip-learned`, docs/PLAN.md task
/// 196), and a chunk with none left is not fed. With `last`, each chunk's
/// output notes each spec's groups' last `withheld_reason` for the run's
/// closing lines (docs/PLAN.md task 116, H).
#[allow(clippy::too_many_arguments)]
fn run_with(
    bank: &mut Bank,
    input: Input<'_>,
    output: Output<'_>,
    closed: Option<(&Path, Format)>,
    skip: Option<&Learned>,
    mut last: Option<&mut LastReasons>,
    opts: RunOptions,
    mut progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<RunStats> {
    let chunk_size = NonZeroUsize::new(opts.chunk_size)
        .ok_or_else(|| polars_err!(ComputeError: "chunk_size must be > 0"))?;
    // The schema of an empty output, when the source turns out to be empty
    // and no frame ever reaches the bank.
    let empty_input = match &input {
        Input::Lazy(lf) => Empty::Plan(Box::new(lf.clone().limit(0))),
        Input::Batches { schema, .. } => Empty::Frame(DataFrame::empty_with_schema(schema)),
    };
    // Where the run's time goes, to stderr when ONLINE_TIMING is set: how
    // long this thread waited for the reader, spent in the bank, and waited
    // for the writer, and how long the writer was busy. The waits are the
    // pipeline's slack -- a run that waits on the reader is read-bound, one
    // that waits on the writer is write-bound (docs/PERFORMANCE.md).
    let timing = std::env::var_os("ONLINE_TIMING").is_some();
    let t_start = Instant::now();
    let mut t_read_wait = Duration::ZERO;
    let mut t_bank = Duration::ZERO;
    let mut t_deliver_wait = Duration::ZERO;

    std::thread::scope(|scope| {
        let (read_tx, read_rx) = sync_channel::<Read>(1);
        scope.spawn(move || match input {
            Input::Lazy(lf) => read_plan(lf, chunk_size, read_tx),
            Input::Batches { frames, .. } => read_frames(frames, read_tx),
        });

        let (write_tx, write_rx) = sync_channel::<Write_>(1);
        let discarding = matches!(output, Output::Discard);
        let (mut sink, writer) = match output {
            Output::File { path, format } => (
                None,
                Some(scope.spawn(move || write_file(path, format, write_rx))),
            ),
            Output::Batches(f) => (Some(f), None),
            // No writer thread and no sink: `deliver` drops the frame.
            Output::Discard => (None, None),
        };
        // The closed groups' writer, fed a drained frame after every chunk.
        let (closed_tx, closed_rx) = sync_channel::<Write_>(1);
        let closed_writer =
            closed.map(|(path, format)| scope.spawn(move || write_file(path, format, closed_rx)));
        let mut closed_sent = false;
        // Hand a frame on, to the writer thread or the caller's callback.
        // A closed writer channel means the writer failed; its own error is
        // the one to report, and `join` below has it.
        let mut deliver = |df: DataFrame| -> PolarsResult<()> {
            match &mut sink {
                Some(f) => f(df),
                None if discarding => Ok(()),
                None => write_tx
                    .send(Write_::Chunk(df))
                    .map_err(|_| polars_err!(ComputeError: "{}", WRITER_STOPPED)),
            }
        };

        let mut stats = RunStats::default();
        let run = || -> PolarsResult<()> {
            loop {
                let t = Instant::now();
                let msg = read_rx.recv();
                t_read_wait += t.elapsed();
                let chunk = match msg {
                    Ok(Read::Chunk(c)) => c,
                    Ok(Read::End(r)) => {
                        r?;
                        break;
                    }
                    // The reader thread is gone without a word: a panic.
                    Err(_) => polars_bail!(ComputeError: "the reader stopped"),
                };
                let chunk = match skip {
                    Some(learned) => {
                        let kept = chunk.filter(&learned.unlearned(&chunk)?)?;
                        if kept.height() == 0 {
                            continue;
                        }
                        kept
                    }
                    None => chunk,
                };
                let height = chunk.height();
                let t = Instant::now();
                // `stats.rows` is the rows fed before this chunk, so an
                // error names the input's row (task 120).
                let out = augment(bank, chunk, opts.predict, opts.learn_only, stats.rows)?;
                if let Some(last) = last.as_deref_mut() {
                    last.note(bank.specs(), &out)?;
                }
                t_bank += t.elapsed();
                let t = Instant::now();
                deliver(out)?;
                t_deliver_wait += t.elapsed();
                if closed_writer.is_some() {
                    let rows = bank
                        .closed_groups(None, true)
                        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
                    if rows.height() > 0 {
                        closed_tx
                            .send(Write_::Chunk(rows))
                            .map_err(|_| polars_err!(ComputeError: "{}", WRITER_STOPPED))?;
                        closed_sent = true;
                    }
                }
                stats.rows += height;
                stats.chunks += 1;
                progress(stats)?;
            }
            if stats.chunks == 0 && !discarding {
                // Empty input: still produce a valid, empty output with the
                // right schema. Nothing to do when there is no output --
                // collecting the plan only to drop it would be a read of the
                // source for no one.
                let empty = match empty_input {
                    Empty::Plan(lf) => lf.collect()?,
                    Empty::Frame(df) => df,
                };
                deliver(augment(bank, empty, opts.predict, opts.learn_only, 0)?)?;
            }
            if closed_writer.is_some() && !closed_sent {
                // Nothing closed: the empty frame with the schema, as an
                // empty output has.
                let rows = bank
                    .closed_groups(None, true)
                    .map_err(|e| polars_err!(ComputeError: "{}", e))?;
                closed_tx
                    .send(Write_::Chunk(rows))
                    .map_err(|_| polars_err!(ComputeError: "{}", WRITER_STOPPED))?;
            }
            Ok(())
        };
        let result = run();
        // Dropping the receiver is what stops a reader still at work: its
        // next `send` fails and the query is told to stop.
        drop(read_rx);
        // The closed groups' file goes the output's way: published by a run
        // that got here in full, discarded by one that did not. Its own
        // failure, when that is what stopped the run, is the one to report.
        if let Some(w) = closed_writer {
            // Published by a run that got here in full, and by one that did
            // not when it drained anything: a drained row has left the bank,
            // and the file is the only place it is -- what
            // `fit_predict_batches` does on a `break` or an error (the second
            // review of 2026-09-15, F2). The output keeps its whole-run rule
            // below.
            if result.is_ok() || closed_sent {
                let _ = closed_tx.send(Write_::End);
            }
            drop(closed_tx);
            let written = w.join().expect("the closed-groups writer thread panicked");
            match (&result, written) {
                (Ok(()), Err(e)) => return Err(e),
                (Err(e), Err(we)) if e.to_string() == WRITER_STOPPED => return Err(we),
                _ => {}
            }
        }
        let t_writer = match (result, writer) {
            (Ok(()), Some(w)) => {
                // Only a run that got here in full publishes the output.
                let _ = write_tx.send(Write_::End);
                drop(write_tx);
                w.join().expect("the writer thread panicked")?
            }
            (Err(e), Some(w)) => {
                drop(write_tx);
                // The writer's failure, if that is what stopped the run, is
                // the one to report; otherwise the run's own.
                match w.join().expect("the writer thread panicked") {
                    Err(we) if e.to_string() == WRITER_STOPPED => return Err(we),
                    _ => return Err(e),
                }
            }
            (r, None) => {
                r?;
                Duration::ZERO
            }
        };
        if timing {
            eprintln!(
                "ONLINE_TIMING run rows={} chunks={} read_wait={:.2}s bank={:.2}s \
                 write_wait={:.2}s writer_busy={:.2}s total={:.2}s",
                stats.rows,
                stats.chunks,
                t_read_wait.as_secs_f64(),
                t_bank.as_secs_f64(),
                t_deliver_wait.as_secs_f64(),
                t_writer.as_secs_f64(),
                t_start.elapsed().as_secs_f64(),
            );
        }
        Ok(stats)
    })
}

/// Each spec's groups' last `withheld_reason` in the run, as the output
/// carried it (docs/PLAN.md task 116, H): the summary keeps no reason per
/// group, so the runner reads it off each chunk it writes, `O(groups)`.
#[derive(Debug, Default)]
struct LastReasons(Vec<std::collections::HashMap<String, Option<String>>>);

impl LastReasons {
    /// Note `out`'s rows, the bank's columns among them, in row order: a
    /// group's last row in the chunk is the one kept. A spec writes its
    /// first instance's `withheld_reason` first; a group is its key's text,
    /// the null key "null", and a spec without a group one key, "".
    fn note(&mut self, specs: &[Spec], out: &DataFrame) -> PolarsResult<()> {
        if self.0.len() != specs.len() {
            self.0.resize_with(specs.len(), Default::default);
        }
        for (si, spec) in specs.iter().enumerate() {
            let Some(col) = out.column(&spec.name).ok() else {
                continue;
            };
            let Ok(st) = col.struct_() else {
                continue;
            };
            let Some(field) = st
                .fields_as_series()
                .into_iter()
                .find(|f| f.name().starts_with("withheld_reason"))
            else {
                continue;
            };
            let reasons = field.cast(&DataType::String)?;
            let reasons = reasons.str()?;
            let keys = match &spec.group {
                Some(g) => Some(
                    out.column(g)?
                        .as_materialized_series()
                        .cast(&DataType::String)?,
                ),
                None => None,
            };
            let keys = keys.as_ref().map(|k| k.str()).transpose()?;
            let map = &mut self.0[si];
            let mut seen = std::collections::HashSet::new();
            for i in (0..out.height()).rev() {
                let key = match keys {
                    Some(k) => k.get(i).unwrap_or("null"),
                    None => "",
                };
                if seen.insert(key) {
                    map.insert(key.to_string(), reasons.get(i).map(str::to_string));
                }
            }
        }
        Ok(())
    }
}

/// The run's closing lines (docs/WARMUP-AND-CONVERGENCE.md §7.9): per spec,
/// the groups whose last row in the run was withheld, counted by reason, and
/// the groups whose smallest data share (`min_support_coef` in the summary)
/// is below 0.5; nothing for a spec with neither.
fn readiness_lines(bank: &Bank, last: &LastReasons) -> Vec<String> {
    let groups = |n: usize| if n == 1 { "group" } else { "groups" };
    let mut lines = Vec::new();
    for (si, spec) in bank.specs().iter().enumerate() {
        let mut by_reason = std::collections::BTreeMap::<&str, usize>::new();
        if let Some(map) = last.0.get(si) {
            for reason in map.values().flatten() {
                *by_reason.entry(reason.as_str()).or_default() += 1;
            }
        }
        let low = bank
            .summary(si, None)
            .ok()
            .and_then(|df| {
                let c = df.column("min_support_coef").ok()?.f64().ok()?.clone();
                Some(c.iter().flatten().filter(|v| *v < 0.5).count())
            })
            .unwrap_or(0);
        let mut parts = Vec::new();
        if !by_reason.is_empty() {
            let n: usize = by_reason.values().sum();
            let each: Vec<String> = by_reason.iter().map(|(r, c)| format!("{r} {c}")).collect();
            parts.push(format!(
                "{n} {} whose last row was withheld ({})",
                groups(n),
                each.join(", ")
            ));
        }
        if low > 0 {
            parts.push(format!("{low} {} with min_support_coef < 0.5", groups(low)));
        }
        if !parts.is_empty() {
            lines.push(format!("spec {:?}: {}", spec.name, parts.join("; ")));
        }
    }
    lines
}

/// The bank's columns appended to `chunk`, aligned for the writers, which
/// walk the columns' chunks in lockstep: an input frame that spans a
/// row-group boundary arrives as several arrow chunks per column, and the
/// bank's columns are one chunk each.
fn augment(
    bank: &mut Bank,
    chunk: DataFrame,
    predict: bool,
    learn_only: bool,
    row_base: usize,
) -> PolarsResult<DataFrame> {
    let cols = if predict {
        bank.predict_from(&chunk, row_base)?
    } else {
        // A run with no output keeps no prediction: `ModelBank.fit`'s
        // run, which takes any embargo (review R2, P4).
        bank.fit_predict_from_with(&chunk, row_base, learn_only)?
    };
    // A readiness notice is a line on stderr, once per (spec, group,
    // instance), as the Python layer warns once (docs/WARMUP-AND-CONVERGENCE.md
    // §3): a coefficient more ridge than data, or a noise gate that cannot
    // be met.
    for notice in bank.take_notices() {
        eprintln!("online: {notice}");
    }
    let mut out = chunk;
    for c in cols {
        out.with_column(c)?;
    }
    out.align_chunks_par();
    Ok(out)
}

/// The empty output's schema, when no frame ever reaches the bank: a plan
/// to collect, or a frame already made.
enum Empty {
    Plan(Box<LazyFrame>),
    Frame(DataFrame),
}

/// Stage 1, for a plan: run `input` on the streaming engine, handing over
/// every `chunk_size` rows in order. Blocks while the engine works, so it
/// gets a thread of its own.
fn read_plan(input: LazyFrame, chunk_size: NonZeroUsize, tx: SyncSender<Read>) {
    let chunks = tx.clone();
    let callback = PlanCallback::new(move |df: DataFrame| {
        // A closed channel is the run giving up; `true` tells the engine to
        // stop.
        Ok(chunks.send(Read::Chunk(df)).is_err())
    });
    let result = input
        .sink_batches(callback, true, Some(chunk_size))
        .and_then(|lf| lf.collect_with_engine(Engine::Streaming))
        .map(|_| ());
    let _ = tx.send(Read::End(result));
}

/// Stage 1, for frames the caller produces: pull them in order until the
/// iterator ends or fails. A closed channel is the run giving up, and the
/// iterator is dropped without being drained.
fn read_frames(
    frames: Box<dyn Iterator<Item = PolarsResult<DataFrame>> + Send + '_>,
    tx: SyncSender<Read>,
) {
    for frame in frames {
        let end = match frame {
            Ok(df) => {
                if tx.send(Read::Chunk(df)).is_err() {
                    return;
                }
                continue;
            }
            Err(e) => Err(e),
        };
        let _ = tx.send(Read::End(end));
        return;
    }
    let _ = tx.send(Read::End(Ok(())));
}

/// Stage 3: write the frames to `path` in `format`, through a temporary
/// that is renamed into place on `End`. A sender that goes away without
/// `End` is a failed run, and the temporary is removed instead. Returns the
/// time spent writing, for the timing line.
fn write_file(path: &Path, format: Format, rx: Receiver<Write_>) -> PolarsResult<Duration> {
    write_frames(path, format, rx).map_err(|e| match e {
        // Polars' writers report the filesystem without the file; name it.
        PolarsError::IO { error, msg } => {
            let inner = msg.map_or_else(|| error.to_string(), |m| m.to_string());
            PolarsError::IO {
                error,
                msg: Some(format!("writing {}: {inner}", path.display()).into()),
            }
        }
        e => e,
    })
}

fn write_frames(path: &Path, format: Format, rx: Receiver<Write_>) -> PolarsResult<Duration> {
    let (file, pending) = AtomicFile::create(path)?;
    let mut buf = BufWriter::new(file);
    let mut busy = Duration::ZERO;
    // The run always sends a frame before `End` -- an empty one for an empty
    // source -- so the first message is what the writer is opened on.
    let first = match rx.recv() {
        Ok(Write_::Chunk(df)) => df,
        Ok(Write_::End) => polars_bail!(ComputeError: "internal: a run with no frames"),
        Err(_) => return Ok(busy),
    };
    let t = Instant::now();
    let mut writer = FormatWriter::open(format, &mut buf, &first)?;
    writer.write(&first)?;
    busy += t.elapsed();
    let mut complete = false;
    for msg in rx {
        match msg {
            Write_::Chunk(df) => {
                let t = Instant::now();
                writer.write(&df)?;
                busy += t.elapsed();
            }
            Write_::End => {
                complete = true;
                break;
            }
        }
    }
    if !complete {
        return Ok(busy);
    }
    let t = Instant::now();
    writer.finish()?;
    buf.flush()?;
    drop(buf);
    // The output is complete, footer and all; publish it under its own name.
    pending.commit()?;
    busy += t.elapsed();
    Ok(busy)
}

type Sink<'a> = &'a mut BufWriter<File>;

/// One of polars' batched writers, over the runner's file.
enum FormatWriter<'a> {
    Parquet(Box<ParquetSink<'a>>),
    Ipc(polars::io::ipc::BatchedWriter<Sink<'a>>),
    Csv(polars::io::csv::write::BatchedWriter<Sink<'a>>),
    /// The file itself: NDJSON has no header or footer (`ndjson_write`).
    Ndjson(Sink<'a>),
}

impl<'a> FormatWriter<'a> {
    /// Open a writer for frames shaped like `first`.
    fn open(format: Format, sink: Sink<'a>, first: &DataFrame) -> PolarsResult<Self> {
        Ok(match format {
            Format::Parquet => FormatWriter::Parquet(Box::new(ParquetSink::open(sink, first)?)),
            Format::Ipc => {
                let schema = first.schema();
                let arrow = schema.to_arrow(CompatLevel::newest());
                let fields = polars_arrow::io::ipc::write::default_ipc_fields(arrow.iter_values());
                FormatWriter::Ipc(IpcWriter::new(sink).batched(schema, fields)?)
            }
            Format::Csv => {
                let flat = csv_flat(first)?;
                FormatWriter::Csv(CsvWriter::new(sink).batched(flat.schema())?)
            }
            Format::Ndjson => FormatWriter::Ndjson(sink),
        })
    }

    fn write(&mut self, df: &DataFrame) -> PolarsResult<()> {
        if df.height() == 0 {
            // Nothing to encode; opening on the empty frame fixed the schema.
            return Ok(());
        }
        match self {
            FormatWriter::Parquet(w) => w.write(df),
            FormatWriter::Ipc(w) => w.write_batch(df),
            FormatWriter::Csv(w) => w.write_batch(&csv_flat(df)?),
            FormatWriter::Ndjson(sink) => ndjson_write(sink, df),
        }
    }

    fn finish(self) -> PolarsResult<()> {
        match self {
            FormatWriter::Parquet(w) => w.writer.finish().map(|_| ()),
            FormatWriter::Ipc(mut w) => w.finish(),
            FormatWriter::Csv(mut w) => w.finish(),
            FormatWriter::Ndjson(_) => Ok(()),
        }
    }
}

/// `df` as JSON lines, by polars' batched NDJSON writer on this thread
/// (docs/IMPROVEMENTS.md C8, docs/PLAN.md task 115 (g)).
///
/// It serialized a slice per thread of polars' pool, which on macOS's
/// system allocator -- the CLI has no allocator of its own -- ran anywhere
/// from 4.8 s to 54 s on 3M rows (2026-09-02): every thread grew and freed
/// multi-megabyte buffers, which that allocator hands to the kernel. Measured
/// again on 2026-09-28, the three-spec example bank over 3M rows: this
/// writer 4.65-4.78 s wall and 1.1-1.2 s system time against the slices'
/// 4.98-5.50 s and 2.1 s, where parquet takes 4.65-4.73 s.
fn ndjson_write(sink: &mut BufWriter<File>, df: &DataFrame) -> PolarsResult<()> {
    polars::io::json::BatchedWriter::new(sink).write_batch(df)
}

/// Polars' batched parquet writer, encoding the columns of each row group in
/// parallel.
///
/// `BatchedWriter::write_batch` builds the page iterators in parallel but
/// *drives* them -- the encoding and the zstd compression, which is where the
/// time goes -- one column after another on the writing thread; at k=20 that
/// serial work took longer than the bank and set the runner's pace. This is
/// what polars' own streaming sink does instead: encode and compress every
/// leaf column to pages on polars' thread pool (`POLARS_MAX_THREADS`, like
/// its readers; the bank's pool is separate), then hand the finished row
/// group to the writer. Same pages, same file; measured in
/// docs/PERFORMANCE.md.
struct ParquetSink<'a> {
    writer: polars::io::parquet::write::BatchedWriter<Sink<'a>>,
    fields: Vec<polars_parquet::write::ParquetType>,
    encodings: Vec<Vec<polars_parquet::write::Encoding>>,
    options: polars_parquet::write::WriteOptions,
}

impl<'a> ParquetSink<'a> {
    fn open(sink: Sink<'a>, first: &DataFrame) -> PolarsResult<Self> {
        use polars::io::parquet::write::{ParquetCompression, get_encodings};
        use polars_parquet::write::{StatisticsOptions, Version, WriteOptions};
        let mut writer = ParquetWriter::new(sink).batched(first.schema())?;
        let fields = writer.parquet_schema().fields().to_vec();
        let arrow = first.schema().to_arrow(CompatLevel::newest());
        let encodings = get_encodings(&arrow).as_ref().to_vec();
        // `ParquetWriter`'s own defaults, so the file is the one
        // `write_batch` would have written.
        let options = WriteOptions {
            statistics: StatisticsOptions::default(),
            version: Version::V1,
            compression: ParquetCompression::default().into(),
            data_page_size: None,
        };
        Ok(Self {
            writer,
            fields,
            encodings,
            options,
        })
    }

    /// Each aligned chunk of `df` as one row group.
    fn write(&mut self, df: &DataFrame) -> PolarsResult<()> {
        use polars_parquet::parquet::error::{ParquetError, ParquetResult};
        use polars_parquet::write::{CompressedPage, Compressor, array_to_columns};
        use rayon::prelude::*;
        let options = self.options;
        for batch in df.iter_chunks(CompatLevel::newest(), false) {
            let rows = batch.len();
            if rows == 0 {
                continue;
            }
            let columns: Vec<Vec<Vec<CompressedPage>>> = polars_core::runtime::THREAD_POOL
                .install(|| {
                    batch
                        .columns()
                        .par_iter()
                        .zip(&self.fields)
                        .zip(&self.encodings)
                        .map(|((array, field), encoding)| {
                            // A nested column (`coef`) is more than one leaf.
                            array_to_columns(array, field.clone(), options, encoding)?
                                .into_iter()
                                .map(|pages| {
                                    let pages = pages.map(|p| {
                                        p.map_err(|e| {
                                            ParquetError::FeatureNotSupported(format!(
                                                "reraised in polars: {e}"
                                            ))
                                        })
                                    });
                                    Compressor::new_from_vec(pages, options.compression, vec![])
                                        .collect::<ParquetResult<Vec<CompressedPage>>>()
                                        .map_err(PolarsError::from)
                                })
                                .collect::<PolarsResult<Vec<_>>>()
                        })
                        .collect::<PolarsResult<_>>()
                })?;
            let leaves: Vec<Vec<CompressedPage>> = columns.into_iter().flatten().collect();
            self.writer.write_row_group(rows as u64, &leaves)?;
        }
        Ok(())
    }
}

/// `df` as CSV can carry it: each struct column flattened into
/// `<name>.<field>` columns, and each numeric list column -- `coef` -- as
/// JSON text (`[1.5,-0.25]`, empty for null), which
/// `str.json_decode(pl.List(pl.Float64))` reads back. Anything else nested
/// has no CSV form, and the error says which column and what to do.
fn csv_flat(df: &DataFrame) -> PolarsResult<DataFrame> {
    let structs: Vec<PlSmallStr> = df
        .columns()
        .iter()
        .filter(|c| matches!(c.dtype(), DataType::Struct(_)))
        .map(|c| c.name().clone())
        .collect();
    let mut flat = if structs.is_empty() {
        df.clone()
    } else {
        df.unnest(structs, Some("."))?
    };
    for i in 0..flat.width() {
        let c = &flat.columns()[i];
        match c.dtype() {
            DataType::List(inner) if inner.is_primitive_numeric() => {
                let text = list_as_json(c)?;
                flat.replace_column(i, text)?;
            }
            dt if dt.is_nested() => polars_bail!(
                ComputeError:
                "csv cannot carry `{}` ({}): write parquet, ipc or ndjson instead",
                c.name(),
                dt
            ),
            _ => {}
        }
    }
    Ok(flat)
}

/// A numeric list column as JSON arrays in a string column, null for null.
/// Rust's `{}` for an `f64` is the shortest text that reads back to the same
/// value, so nothing is lost on the way through.
fn list_as_json(c: &Column) -> PolarsResult<Column> {
    let ca = c.list()?;
    let mut out = StringChunkedBuilder::new(c.name().clone(), ca.len());
    let mut text = String::new();
    for row in ca.amortized_iter() {
        match row {
            None => out.append_null(),
            Some(s) => {
                text.clear();
                text.push('[');
                let s = s.as_ref().cast(&DataType::Float64)?;
                for (j, v) in s.f64()?.iter().enumerate() {
                    if j > 0 {
                        text.push(',');
                    }
                    match v {
                        Some(v) if v.is_finite() => text.push_str(&format!("{v}")),
                        // JSON has no NaN or infinity; null is the nearest.
                        _ => text.push_str("null"),
                    }
                }
                text.push(']');
                out.append_value(&text);
            }
        }
    }
    Ok(out.finish().into_column())
}
