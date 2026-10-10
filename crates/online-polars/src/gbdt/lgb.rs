//! A LightGBM `Dataset` built from frames through its C API
//! (docs/GBDT-ADAPTER.md §3.2, §6).
//!
//! LightGBM places each feature's bins from a sample of the rows, drawn by
//! `LGBM_SampleIndices` from the row count and `data_random_seed`, and then
//! pushes every value into its bin. Streamed, that is two passes: the first
//! gathers the sampled rows and builds the bins
//! (`LGBM_DatasetCreateFromSampledColumn`), the second pushes each segment
//! as float32 rows built in parallel (`LGBM_DatasetPushRows`, itself
//! parallel over rows), and the label and weight are set at the end. A
//! validation set takes its training set's bins
//! (`LGBM_DatasetCreateByReference`) and needs the push pass alone.
//!
//! In memory, every segment's features go in as one Arrow stream
//! (`LGBM_DatasetCreateFromArrowStream`, LightGBM 4.7.0 on), which LightGBM
//! samples the same way and pushes column by column; the columns are cast
//! to float32 first, so both routes hand LightGBM the same values and grow
//! the same trees. LightGBM's bins are float64 and its own route from a
//! float64 array reads every value as float64; a float32 value lands in
//! another bin only where the rounding crosses a bin edge
//! (docs/GBDT-ADAPTER.md §4.5: validation RMSE equal to six places).

use std::ffi::{CString, c_char, c_int, c_void};
use std::ptr::null_mut;
use std::time::Instant;

use polars::prelude::PlSmallStr;
use polars_arrow::array::{Array, StructArray};
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::ffi::export_iterator;
use rayon::prelude::*;

use super::frames::{Frames, Reader, Segment, checked_f64, segments_of};
use super::libs::{Handle, LgbLib};
use super::{GbdtError, Layout, conversion_pool};

/// `C_API_DTYPE_FLOAT32` (`include/LightGBM/c_api.h`).
const DTYPE_FLOAT32: c_int = 0;
/// LightGBM's `kZeroThreshold` (`include/LightGBM/meta.h`, `1e-35f`): a
/// sampled value at or below it in magnitude counts as zero and is left out
/// of the sample, as `LGBM_DatasetCreateFromMat` leaves it out.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;
/// Rows a conversion task fills at once: 4,096 rows of 51 float32 values is
/// 0.8 MB, inside a core's cache.
const BLOCK: usize = 4096;

/// How to build a dataset.
#[derive(Debug, Clone)]
pub struct LgbOptions {
    /// LightGBM's parameters for the construction, as its `key=value`
    /// string: `max_bin`, `categorical_feature`, `num_threads` and any other
    /// dataset parameter. Training must not change the dataset ones.
    pub params: String,
    /// Threads for the conversions; 0 for all of them.
    pub threads: usize,
    /// The source's row count, which LightGBM's sample is drawn from.
    pub rows: u64,
    /// The whole stream as Arrow, held in memory, rather than two passes.
    pub in_memory: bool,
    /// Frames the reader keeps ready ahead of LightGBM.
    pub read_ahead: usize,
}

/// A built dataset: a handle LightGBM made, which the caller now owns and
/// must free with `LGBM_DatasetFree` once (the Python `Dataset` it is
/// handed to does), and what the build did.
#[derive(Debug, Clone)]
pub struct LgbBuilt {
    pub handle: usize,
    pub rows: u64,
    pub passes: usize,
    pub in_memory: bool,
    /// Seconds gathering the sample and placing the bins (0 without).
    pub sample_s: f64,
    /// Seconds converting and pushing the rows, or building from Arrow.
    pub build_s: f64,
    /// Seconds spent waiting for a frame.
    pub waited_s: f64,
}

/// A dataset handle freed when dropped, unless released.
struct Owned<'a> {
    lib: &'a LgbLib,
    h: Handle,
}

impl Owned<'_> {
    fn release(mut self) -> Handle {
        std::mem::replace(&mut self.h, null_mut())
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        if !self.h.is_null() {
            // SAFETY: a handle LightGBM made, freed once.
            unsafe { (self.lib.free)(self.h) };
        }
    }
}

fn cstr(s: &str) -> Result<CString, GbdtError> {
    CString::new(s).map_err(|_| GbdtError::msg(format!("`{s}` holds a NUL character")))
}

/// The label and weight of every row, in order, as LightGBM stores them
/// (float32).
struct Meta {
    labels: Vec<f32>,
    weights: Option<Vec<f32>>,
}

impl Meta {
    fn new(layout: &Layout, n: usize) -> Self {
        Self {
            labels: Vec::with_capacity(n),
            weights: layout.weight.as_ref().map(|_| Vec::with_capacity(n)),
        }
    }

    fn push(&mut self, seg: &Segment, layout: &Layout, row: u64) -> Result<(), GbdtError> {
        let p = layout.names.len();
        let lab = checked_f64(seg, p, layout, row)?;
        self.labels.extend(lab.iter().map(|&v| v as f32));
        if let Some(w) = self.weights.as_mut() {
            let vals = checked_f64(seg, p + 1, layout, row)?;
            w.extend(vals.iter().map(|&v| v as f32));
        }
        Ok(())
    }

    fn set(&self, lib: &LgbLib, ds: Handle) -> Result<(), GbdtError> {
        let field = |name: &str, vals: &[f32]| -> Result<(), GbdtError> {
            let f = cstr(name)?;
            // SAFETY: `vals` holds `len` float32 values; LightGBM copies them.
            let rc = unsafe {
                (lib.set_field)(
                    ds,
                    f.as_ptr(),
                    vals.as_ptr().cast(),
                    vals.len() as c_int,
                    DTYPE_FLOAT32,
                )
            };
            lib.check(rc, &format!("setting the {name}"))
        };
        field("label", &self.labels)?;
        if let Some(w) = &self.weights {
            field("weight", w)?;
        }
        Ok(())
    }
}

fn set_names(lib: &LgbLib, ds: Handle, layout: &Layout) -> Result<(), GbdtError> {
    let names = layout
        .names
        .iter()
        .map(|n| cstr(n))
        .collect::<Result<Vec<_>, _>>()?;
    let ptrs: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
    // SAFETY: a live dataset; the strings outlive the call, which copies them.
    let rc = unsafe { (lib.set_names)(ds, ptrs.as_ptr(), ptrs.len() as c_int) };
    lib.check(rc, "setting the feature names")
}

/// A segment's features as row-major float32, NaN for a null, built in
/// parallel over blocks of rows into `buf`.
fn rows_f32(seg: &Segment, p: usize, buf: &mut Vec<f32>, pool: &rayon::ThreadPool) {
    buf.clear();
    buf.resize(seg.n * p, 0.0);
    pool.install(|| {
        buf.par_chunks_mut(BLOCK * p)
            .enumerate()
            .for_each(|(bi, chunk)| {
                let r0 = bi * BLOCK;
                for j in 0..p {
                    seg.num(j).fill_f32(chunk, p, j, r0);
                }
            });
    });
}

fn rows_i32(n: u64) -> Result<i32, GbdtError> {
    i32::try_from(n).map_err(|_| {
        GbdtError::msg(format!(
            "{n} rows: LightGBM's C API counts rows in a 32-bit integer, so a dataset holds \
             {} at most",
            i32::MAX
        ))
    })
}

fn wrong_rows(expected: u64, got: u64) -> GbdtError {
    GbdtError::msg(format!(
        "the source was counted at {expected} rows and a pass over it gave {got}: LightGBM \
         reads it more than once, and every pass must give the same rows"
    ))
}

/// A `Dataset` of the frames, binned on `reference`'s bins when given (a
/// validation set on its training set's), otherwise on LightGBM's own
/// sample.
pub fn lightgbm_dataset(
    lib: &LgbLib,
    frames: Box<dyn Frames>,
    layout: &Layout,
    reference: Option<usize>,
    opts: &LgbOptions,
) -> Result<LgbBuilt, GbdtError> {
    layout.check()?;
    if opts.rows == 0 {
        return Err(GbdtError::msg("the source has no rows"));
    }
    let n = rows_i32(opts.rows)?;
    let pool = conversion_pool(opts.threads)?;
    let params = cstr(&opts.params)?;
    let mut reader = Reader::new(frames, opts.read_ahead);
    let built = if opts.in_memory {
        let Some(from_arrow) = lib.from_arrow else {
            return Err(GbdtError::msg(
                "in_memory=True hands LightGBM an Arrow stream, which it takes from 4.7.0 on",
            ));
        };
        arrow_route(
            lib,
            from_arrow,
            &mut reader,
            layout,
            reference,
            &params,
            n,
            &pool,
        )?
    } else {
        rows_route(lib, &mut reader, layout, reference, &params, n, &pool)?
    };
    let (ds, sample_s, build_s) = built;
    let mut got: c_int = 0;
    // SAFETY: a live dataset.
    lib.check(
        unsafe { (lib.num_data)(ds.h, &mut got) },
        "counting the rows",
    )?;
    Ok(LgbBuilt {
        handle: ds.release() as usize,
        rows: got as u64,
        passes: reader.passes,
        in_memory: opts.in_memory,
        sample_s,
        build_s,
        waited_s: reader.waited.as_secs_f64(),
    })
}

type Built<'a> = (Owned<'a>, f64, f64);

/// Two passes: LightGBM's sample, then the pushes.
fn rows_route<'a>(
    lib: &'a LgbLib,
    reader: &mut Reader,
    layout: &Layout,
    reference: Option<usize>,
    params: &CString,
    n: i32,
    pool: &rayon::ThreadPool,
) -> Result<Built<'a>, GbdtError> {
    let p = layout.names.len();
    let t = Instant::now();
    let ds = match reference {
        Some(r) => {
            let mut h: Handle = null_mut();
            // SAFETY: `r` is a live dataset (the Python `Dataset` holding it
            // is alive for the call); a fresh handle for LightGBM to fill.
            lib.check(
                unsafe { (lib.by_reference)(r as Handle, i64::from(n), &mut h) },
                "creating the dataset on its reference's bins",
            )?;
            Owned { lib, h }
        }
        None => sampled(lib, reader, layout, params, n, pool)?,
    };
    let sample_s = if reference.is_some() {
        0.0
    } else {
        t.elapsed().as_secs_f64()
    };
    reader.restart();
    let t = Instant::now();
    let mut meta = Meta::new(layout, n as usize);
    let mut buf: Vec<f32> = Vec::new();
    let mut start: u64 = 0;
    while let Some(df) = reader.next()? {
        for seg in segments_of(df, layout)? {
            if start + seg.n as u64 > n as u64 {
                return Err(wrong_rows(n as u64, start + seg.n as u64));
            }
            meta.push(&seg, layout, start)?;
            rows_f32(&seg, p, &mut buf, pool);
            // SAFETY: `buf` holds `seg.n * p` float32 values, row-major.
            let rc = unsafe {
                (lib.push_rows)(
                    ds.h,
                    buf.as_ptr().cast::<c_void>(),
                    DTYPE_FLOAT32,
                    seg.n as i32,
                    p as i32,
                    start as i32,
                )
            };
            lib.check(rc, "pushing rows")?;
            start += seg.n as u64;
        }
    }
    if start != n as u64 {
        return Err(wrong_rows(n as u64, start));
    }
    meta.set(lib, ds.h)?;
    set_names(lib, ds.h, layout)?;
    Ok((ds, sample_s, t.elapsed().as_secs_f64()))
}

/// LightGBM's sample rows gathered in one pass, and the bins placed from
/// them: what `LGBM_DatasetCreateFromMat` does with a matrix in memory.
fn sampled<'a>(
    lib: &'a LgbLib,
    reader: &mut Reader,
    layout: &Layout,
    params: &CString,
    n: i32,
    pool: &rayon::ThreadPool,
) -> Result<Owned<'a>, GbdtError> {
    let p = layout.names.len();
    let mut cnt: c_int = 0;
    // SAFETY: an integer for LightGBM to fill.
    lib.check(
        unsafe { (lib.sample_count)(n, params.as_ptr(), &mut cnt) },
        "sizing the sample",
    )?;
    let mut idx = vec![0i32; cnt.max(0) as usize];
    let mut got: i32 = 0;
    // SAFETY: `idx` holds the `cnt` indices LightGBM said it would write.
    lib.check(
        unsafe { (lib.sample_indices)(n, params.as_ptr(), idx.as_mut_ptr().cast(), &mut got) },
        "drawing the sample",
    )?;
    idx.truncate(got.max(0) as usize);
    // The sample's order is the order LightGBM gave; its values are found
    // in stream order.
    let mut order: Vec<(i32, usize)> = idx.iter().copied().zip(0..).collect();
    order.sort_unstable();
    let mut sv: Vec<Vec<f64>> = vec![Vec::new(); p];
    let mut si: Vec<Vec<usize>> = vec![Vec::new(); p];
    let mut row: u64 = 0;
    let mut k = 0usize;
    while let Some(df) = reader.next()? {
        for seg in segments_of(df, layout)? {
            let end = row + seg.n as u64;
            let lo = k;
            while k < order.len() && (order[k].0 as u64) < end {
                k += 1;
            }
            let picks = &order[lo..k];
            if !picks.is_empty() {
                pool.install(|| {
                    sv.par_iter_mut()
                        .zip(si.par_iter_mut())
                        .enumerate()
                        .for_each(|(j, (vals, at))| {
                            let num = seg.num(j);
                            for &(i, s) in picks {
                                let x = f64::from(num.f32_at((i as u64 - row) as usize));
                                if x.abs() > ZERO_THRESHOLD || x.is_nan() {
                                    vals.push(x);
                                    at.push(s);
                                }
                            }
                        });
                });
            }
            row = end;
        }
    }
    if row != n as u64 {
        return Err(wrong_rows(n as u64, row));
    }
    // Each column's values in the sample's own order, as LightGBM's
    // `CreateFromMat` lists them.
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut ats: Vec<Vec<c_int>> = Vec::with_capacity(p);
    for (v, a) in sv.into_iter().zip(si) {
        let mut pairs: Vec<(usize, f64)> = a.into_iter().zip(v).collect();
        pairs.sort_unstable_by_key(|x| x.0);
        ats.push(pairs.iter().map(|x| x.0 as c_int).collect());
        vals.push(pairs.into_iter().map(|x| x.1).collect());
    }
    let mut vptr: Vec<*mut f64> = vals.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let mut iptr: Vec<*mut c_int> = ats.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let per: Vec<c_int> = vals.iter().map(|v| v.len() as c_int).collect();
    let mut h: Handle = null_mut();
    // SAFETY: the column arrays stay alive across the call, which copies
    // what it keeps.
    let rc = unsafe {
        (lib.from_sampled)(
            vptr.as_mut_ptr(),
            iptr.as_mut_ptr(),
            p as i32,
            per.as_ptr(),
            got,
            n,
            i64::from(n),
            params.as_ptr(),
            &mut h,
        )
    };
    lib.check(rc, "placing the bins from the sample")?;
    Ok(Owned { lib, h })
}

type FromArrow = unsafe extern "C" fn(*mut c_void, *const c_char, Handle, *mut Handle) -> c_int;

/// One pass, every segment held as float32 Arrow, handed over whole.
#[allow(clippy::too_many_arguments)]
fn arrow_route<'a>(
    lib: &'a LgbLib,
    from_arrow: FromArrow,
    reader: &mut Reader,
    layout: &Layout,
    reference: Option<usize>,
    params: &CString,
    n: i32,
    pool: &rayon::ThreadPool,
) -> Result<Built<'a>, GbdtError> {
    let p = layout.names.len();
    let t = Instant::now();
    let fields: Vec<Field> = layout
        .names
        .iter()
        .map(|name| {
            Field::new(
                PlSmallStr::from(name.as_str()),
                ArrowDataType::Float32,
                true,
            )
        })
        .collect();
    let dtype = ArrowDataType::Struct(fields);
    let mut meta = Meta::new(layout, n as usize);
    let mut chunks: Vec<Box<dyn Array>> = Vec::new();
    let mut row: u64 = 0;
    while let Some(df) = reader.next()? {
        for seg in segments_of(df, layout)? {
            meta.push(&seg, layout, row)?;
            let cols: Vec<Box<dyn Array>> = pool.install(|| {
                (0..p)
                    .into_par_iter()
                    .map(|j| seg.num(j).to_f32().boxed())
                    .collect()
            });
            chunks.push(StructArray::new(dtype.clone(), seg.n, cols, None).boxed());
            row += seg.n as u64;
        }
    }
    if row != n as u64 {
        return Err(wrong_rows(n as u64, row));
    }
    let field = Field::new(PlSmallStr::EMPTY, dtype, false);
    let mut stream = export_iterator(Box::new(chunks.into_iter().map(Ok)), field);
    let mut h: Handle = null_mut();
    let refh = reference.map_or(null_mut(), |r| r as Handle);
    // SAFETY: `stream` is a valid Arrow C stream; LightGBM reads it to its
    // end and releases it, after which dropping it does nothing.
    let rc = unsafe { from_arrow((&raw mut stream).cast(), params.as_ptr(), refh, &mut h) };
    drop(stream);
    lib.check(rc, "building the dataset from Arrow")?;
    let ds = Owned { lib, h };
    meta.set(lib, ds.h)?;
    set_names(lib, ds.h, layout)?;
    Ok((ds, 0.0, t.elapsed().as_secs_f64()))
}
