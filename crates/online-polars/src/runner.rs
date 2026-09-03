//! Streaming runner shared by the CLI and `polars_online.run` (docs/PLAN.md
//! §11 task 15; docs/ENHANCEMENTS.md E32).
//!
//! Any source polars can scan comes in, any file format polars can write goes
//! out, and the numbers in between are the bank's. A run is a three-stage
//! pipeline:
//!
//! 1. a reader thread hands over frames in stream order: polars' streaming
//!    engine reading a plan in `chunk_rows` rows (`sink_batches`), or an
//!    iterator of frames the caller already has ([`Input::Batches`] -- the
//!    Python API reads with py-polars and feeds the frames in here);
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
use crate::bank::Bank;
use crate::spec::Spec;

/// Writing the output through a temporary is filesystem work, and polars'
/// error type is what the runner returns: its `IO` variant, kind intact, so
/// a caller can tell a file that could not be written or read from a run
/// that was refused (the Python `run` raises `OSError` -- the subclass the
/// kind names -- for the one and `ValueError` for the other). `what` and
/// `path` say which file: the `io::Error` alone does not.
fn io_err(what: &str, path: &Path, e: std::io::Error) -> PolarsError {
    let msg = format!("{what} {}: {e}", path.display());
    PolarsError::IO {
        error: Arc::new(e),
        msg: Some(msg.into()),
    }
}

/// The directory `path` would be written into, or the error naming it: a
/// missing directory is reported before a run, not after the stream it would
/// have cost. `what` is as for [`io_err`].
fn check_parent(what: &str, path: &Path) -> PolarsResult<()> {
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

/// A run description, deserialized from TOML by the CLI and from JSON by
/// `polars_online.run`. An unknown key is refused, naming the keys there
/// are, so a misspelt one cannot silently fall back to its default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Input path. Empty when the caller supplies the source itself
    /// ([`run_config_on`]), which is how `polars_online.run` passes a
    /// `LazyFrame`.
    #[serde(default)]
    pub input: PathBuf,
    /// Output path.
    pub output: PathBuf,
    /// How to read `input`; its extension decides when unset.
    #[serde(default)]
    pub input_format: Option<Format>,
    /// How to write `output`; its extension decides when unset.
    #[serde(default)]
    pub output_format: Option<Format>,
    /// Rows per chunk, [`DEFAULT_CHUNK_ROWS`] when unset. Chunking never
    /// changes the numbers (docs/PLAN.md §9 class 2); it only trades memory
    /// for overhead.
    #[serde(default = "default_chunk_rows")]
    pub chunk_rows: usize,
    /// Load the bank state from here before running (resume).
    #[serde(default)]
    pub load_state: Option<PathBuf>,
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
    /// The model specs to run.
    pub specs: Vec<Spec>,
}

/// Rows per chunk when a config does not say: enough to amortize the
/// per-chunk work, small enough to keep three frames of it in memory.
pub const DEFAULT_CHUNK_ROWS: usize = 100_000;

fn default_chunk_rows() -> usize {
    DEFAULT_CHUNK_ROWS
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub rows: usize,
    pub chunks: usize,
}

impl RunConfig {
    /// Everything but the input, which [`run_config`] checks and
    /// [`run_config_on`] does not need.
    pub fn validate(&self) -> Result<(), String> {
        if self.specs.is_empty() {
            return Err("config has no [[specs]] entries".into());
        }
        if self.chunk_rows == 0 {
            return Err("chunk_rows must be > 0".into());
        }
        if self.output.as_os_str().is_empty() {
            return Err("output is required".into());
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
        self.output_format()?;
        for s in &self.specs {
            s.validate()?;
        }
        Ok(())
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
    /// `chunk_rows` rows: a scan, a query, an in-memory frame's `lazy()`.
    Lazy(LazyFrame),
    /// Frames the caller produces, in stream order and in whatever sizes it
    /// has them (`chunk_rows` does not re-chunk them). `schema` is the frames'
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
}

/// The knobs of [`run`] that are not the source or the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOptions {
    /// Rows per chunk. See [`RunConfig::chunk_rows`].
    pub chunk_rows: usize,
    /// Score instead of learn. See [`RunConfig::predict`].
    pub predict: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            chunk_rows: default_chunk_rows(),
            predict: false,
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
/// cannot be read, and for a `save_state` whose directory is not there,
/// checked before the run because finding out after it would leave the
/// output written and the state lost; `ComputeError` as `loading state
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
    cfg.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let format = cfg
        .input_format()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    // `keep_columns` is applied once, by `run_config_on`.
    run_config_on(cfg, Input::Lazy(format.scan(&cfg.input)?), progress)
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
    cfg.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let format = cfg
        .output_format()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
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
    let mut bank = cfg.open_bank()?;
    let opts = RunOptions {
        chunk_rows: cfg.chunk_rows,
        predict: cfg.predict,
    };
    let stats = run(
        &mut bank,
        input,
        Output::File {
            path: &cfg.output,
            format,
        },
        opts,
        progress,
    )?;
    if let Some(p) = &cfg.save_state {
        bank.save(p).map_err(|e| io_err("saving state", p, e))?;
    }
    Ok(stats)
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
/// it accepted, and a file output is not published.
pub fn run(
    bank: &mut Bank,
    input: Input<'_>,
    output: Output<'_>,
    opts: RunOptions,
    mut progress: impl FnMut(RunStats) -> PolarsResult<()>,
) -> PolarsResult<RunStats> {
    let chunk_rows = NonZeroUsize::new(opts.chunk_rows)
        .ok_or_else(|| polars_err!(ComputeError: "chunk_rows must be > 0"))?;
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
            Input::Lazy(lf) => read_plan(lf, chunk_rows, read_tx),
            Input::Batches { frames, .. } => read_frames(frames, read_tx),
        });

        let (write_tx, write_rx) = sync_channel::<Write_>(1);
        let (mut sink, writer) = match output {
            Output::File { path, format } => (
                None,
                Some(scope.spawn(move || write_file(path, format, write_rx))),
            ),
            Output::Batches(f) => (Some(f), None),
        };
        // Hand a frame on, to the writer thread or the caller's callback.
        // A closed writer channel means the writer failed; its own error is
        // the one to report, and `join` below has it.
        let mut deliver = |df: DataFrame| -> PolarsResult<()> {
            match &mut sink {
                Some(f) => f(df),
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
                let height = chunk.height();
                let t = Instant::now();
                let out = augment(bank, chunk, opts.predict)?;
                t_bank += t.elapsed();
                let t = Instant::now();
                deliver(out)?;
                t_deliver_wait += t.elapsed();
                stats.rows += height;
                stats.chunks += 1;
                progress(stats)?;
            }
            if stats.chunks == 0 {
                // Empty input: still produce a valid, empty output with the
                // right schema.
                let empty = match empty_input {
                    Empty::Plan(lf) => lf.collect()?,
                    Empty::Frame(df) => df,
                };
                deliver(augment(bank, empty, opts.predict)?)?;
            }
            Ok(())
        };
        let result = run();
        // Dropping the receiver is what stops a reader still at work: its
        // next `send` fails and the query is told to stop.
        drop(read_rx);
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

/// The bank's columns appended to `chunk`, aligned for the writers, which
/// walk the columns' chunks in lockstep: an input frame that spans a
/// row-group boundary arrives as several arrow chunks per column, and the
/// bank's columns are one chunk each.
fn augment(bank: &mut Bank, chunk: DataFrame, predict: bool) -> PolarsResult<DataFrame> {
    let cols = if predict {
        bank.predict(&chunk)?
    } else {
        bank.fit_predict(&chunk)?
    };
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
/// every `chunk_rows` rows in order. Blocks while the engine works, so it
/// gets a thread of its own.
fn read_plan(input: LazyFrame, chunk_rows: NonZeroUsize, tx: SyncSender<Read>) {
    let chunks = tx.clone();
    let callback = PlanCallback::new(move |df: DataFrame| {
        // A closed channel is the run giving up; `true` tells the engine to
        // stop.
        Ok(chunks.send(Read::Chunk(df)).is_err())
    });
    let result = input
        .sink_batches(callback, true, Some(chunk_rows))
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
    /// The file itself: NDJSON has no header or footer, and each frame is
    /// serialized in slices on the thread pool (`ndjson_write`).
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

/// `df` as JSON lines, serialized a slice per thread and written in order --
/// the same serializer as polars' batched NDJSON writer, which runs it over
/// the whole frame on one thread and took five times as long as the bank.
fn ndjson_write(sink: &mut BufWriter<File>, df: &DataFrame) -> PolarsResult<()> {
    use rayon::prelude::*;
    let rows = df.height();
    let per = rows.div_ceil(rayon::current_num_threads().max(1)).max(1024);
    let parts: Vec<Vec<u8>> = (0..rows)
        .step_by(per)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let mut buf = Vec::new();
            polars::io::json::BatchedWriter::new(&mut buf)
                .write_batch(&df.slice(start as i64, per))?;
            Ok(buf)
        })
        .collect::<PolarsResult<_>>()?;
    for part in parts {
        sink.write_all(&part)?;
    }
    Ok(())
}

/// Polars' batched parquet writer, encoding the columns of each row group in
/// parallel.
///
/// `BatchedWriter::write_batch` builds the page iterators in parallel but
/// *drives* them -- the encoding and the zstd compression, which is where the
/// time goes -- one column after another on the writing thread; at k=20 that
/// serial work took longer than the bank and set the runner's pace. This is
/// what polars' own streaming sink does instead: encode and compress every
/// leaf column to pages on the thread pool, then hand the finished row group
/// to the writer. Same pages, same file; measured in docs/PERFORMANCE.md.
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
            let columns: Vec<Vec<Vec<CompressedPage>>> = batch
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
                .collect::<PolarsResult<_>>()?;
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
