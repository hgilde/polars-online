//! Streaming parquet runner shared by the CLI (docs/PLAN.md §11 task 15).
//!
//! Reads a parquet file in row-group-sized batches, feeds each batch to a
//! [`Bank`], and writes the augmented batches out — so memory stays O(state)
//! rather than O(data). Supports resuming from and saving to a bank state file.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use polars::prelude::*;
use polars_utils::pl_path::PlRefPath;
use serde::{Deserialize, Serialize};

use crate::atomic::AtomicFile;
use crate::bank::Bank;
use crate::spec::Spec;

/// Writing the output through a temporary is filesystem work, and polars'
/// error type is what the runner returns.
fn io_err(e: std::io::Error) -> PolarsError {
    polars_err!(ComputeError: "{}", e)
}

/// A run description, deserialized from TOML by the CLI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Input parquet path.
    pub input: PathBuf,
    /// Output parquet path.
    pub output: PathBuf,
    /// Rows per chunk. Chunking never changes the numbers (docs/PLAN.md §9
    /// class 2); it only trades memory for overhead.
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
    /// The model specs to run.
    pub specs: Vec<Spec>,
}

fn default_chunk_rows() -> usize {
    100_000
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    pub rows: usize,
    pub chunks: usize,
}

impl RunConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.specs.is_empty() {
            return Err("config has no [[specs]] entries".into());
        }
        if self.chunk_rows == 0 {
            return Err("chunk_rows must be > 0".into());
        }
        for s in &self.specs {
            s.validate()?;
        }
        Ok(())
    }
}

/// Run a config end to end. `progress` is called after each chunk with the
/// running stats, so the CLI can print without this crate knowing about stdout.
pub fn run_config(cfg: &RunConfig, mut progress: impl FnMut(RunStats)) -> PolarsResult<RunStats> {
    cfg.validate()
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let mut bank = match &cfg.load_state {
        Some(p) => Bank::load(p, Some(&cfg.specs))
            .map_err(|e| polars_err!(ComputeError: "loading state {}: {}", p.display(), e))?,
        None => Bank::new(cfg.specs.clone()).map_err(|e| polars_err!(ComputeError: "{}", e))?,
    };

    let input = PlRefPath::try_from_pathbuf(cfg.input.clone())?;
    let mut lf = LazyFrame::scan_parquet(input, ScanArgsParquet::default())?;
    if !cfg.keep_columns.is_empty() {
        let cols: Vec<Expr> = cfg.keep_columns.iter().map(|c| col(c.as_str())).collect();
        lf = lf.select(cols);
    }

    // One BatchedWriter for the whole run: each chunk becomes a row group, so
    // the output is written incrementally and memory stays O(chunk).
    let mut writer: Option<polars::io::parquet::write::BatchedWriter<BufWriter<File>>> = None;
    let mut stats = RunStats::default();

    // Read chunk n+1 while chunk n is being fitted and written
    // (docs/PERFORMANCE.md P6). The channel holds one chunk, so the reader
    // stays exactly one ahead and memory is still O(chunk) rather than O(data).
    // Order is preserved by construction -- a single reader on a FIFO -- which
    // matters because chunks must reach the bank in stream order.
    let (tx, rx) = std::sync::mpsc::sync_channel::<PolarsResult<DataFrame>>(1);
    let reader_lf = lf.clone();
    let chunk_rows = cfg.chunk_rows;
    let reader = std::thread::spawn(move || {
        let mut offset: i64 = 0;
        loop {
            match reader_lf
                .clone()
                .slice(offset, chunk_rows as IdxSize)
                .collect()
            {
                Ok(df) => {
                    let height = df.height();
                    if height == 0 || tx.send(Ok(df)).is_err() || height < chunk_rows {
                        break;
                    }
                    offset += height as i64;
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    });

    let mut pending: Option<AtomicFile> = None;
    let mut result: PolarsResult<()> = Ok(());
    for chunk in rx {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                result = Err(e);
                break;
            }
        };
        if chunk.height() == 0 {
            break;
        }
        let height = chunk.height();
        let cols = match bank.fit_predict(&chunk) {
            Ok(c) => c,
            Err(e) => {
                result = Err(e);
                break;
            }
        };
        let mut out = chunk;
        for c in cols {
            out.with_column(c)?;
        }
        let w = match &mut writer {
            Some(w) => w,
            None => {
                // Written to a temporary and renamed into place at the end
                // (`crate::atomic`), so a run that fails halfway leaves the
                // previous output where it was instead of a headless parquet
                // under its name. `AtomicFile`'s `Drop` removes the
                // temporary on the way out.
                let (file, p) = AtomicFile::create(&cfg.output).map_err(io_err)?;
                pending = Some(p);
                let schema = out.schema();
                writer = Some(ParquetWriter::new(BufWriter::new(file)).batched(schema)?);
                writer.as_mut().unwrap()
            }
        };
        w.write_batch(&out)?;

        stats.rows += height;
        stats.chunks += 1;
        progress(stats);
    }
    // Drop the receiver first so a reader still blocked on `send` wakes up.
    let _ = reader.join();
    result?;
    if let Some(w) = writer {
        w.finish()?;
    } else {
        // Empty input: still produce a valid, empty output with the right schema.
        let mut empty = lf.clone().limit(0).collect()?;
        let cols = bank.fit_predict(&empty)?;
        for c in cols {
            empty.with_column(c)?;
        }
        let (file, p) = AtomicFile::create(&cfg.output).map_err(io_err)?;
        pending = Some(p);
        ParquetWriter::new(BufWriter::new(file)).finish(&mut empty)?;
    }
    // The output is complete, footer and all; publish it under its own name.
    if let Some(p) = pending {
        p.commit().map_err(io_err)?;
    }

    if let Some(p) = &cfg.save_state {
        bank.save(p)
            .map_err(|e| polars_err!(ComputeError: "saving state {}: {}", p.display(), e))?;
    }
    Ok(stats)
}
