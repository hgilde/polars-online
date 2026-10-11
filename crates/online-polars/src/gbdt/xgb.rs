//! An XGBoost `QuantileDMatrix` built from frames through its C API
//! (docs/GBDT-ADAPTER.md §3.1, §6).
//!
//! `XGQuantileDMatrixCreateFromCallback` pulls the frames through two
//! callbacks, `reset` and `next`, and reads each batch from a proxy matrix.
//! Each segment of a frame goes to the proxy as a list of
//! `__array_interface__`s on its columns' own Arrow buffers, a column's
//! validity bitmap as its `mask`, so nothing is copied: XGBoost reads each
//! value as float32, a null or a NaN as missing. The label and the weight go
//! in the same way, and the feature names and types with every batch, as
//! xgboost's Python `DataIter` sends them.
//!
//! XGBoost reads the stream three times with a `ref` (counting, indexing,
//! the column matrix) and four without, the extra pass sketching each
//! feature's quantiles over every row. With `sample_rows`, one pass of this
//! module's own draws a sample ([`Sampler`]), weighted when the rows are,
//! and builds a small matrix whose columns plant the cuts, which goes to
//! XGBoost as the `ref`.
//! A validation matrix takes the training matrix as its `ref` instead.

use std::collections::VecDeque;
use std::ffi::{CString, c_char, c_int};
use std::fmt::Write as _;
use std::panic::AssertUnwindSafe;
use std::ptr::null_mut;
use std::time::Instant;

use super::frames::{Frames, Reader, Segment, checked_f64, panic_message, segments_of};
use super::libs::{Handle, XgbLib};
use super::sample::Sampler;
use super::{GbdtError, Layout, conversion_pool};

/// How to build a matrix.
#[derive(Debug, Clone)]
pub struct XgbOptions {
    /// Bins per feature, as XGBoost's `max_bin`; training must use the same.
    pub max_bin: usize,
    /// Threads for XGBoost and for the conversions; 0 for all of them.
    pub threads: usize,
    /// Rows of the sample the cuts come from; `None` lets XGBoost sketch
    /// every row. Ignored when a `ref` is given.
    pub sample_rows: Option<usize>,
    /// The sample's seed.
    pub seed: u64,
    /// Frames the reader keeps ready ahead of XGBoost.
    pub read_ahead: usize,
}

/// A built matrix: a handle XGBoost made, which the caller now owns and
/// must free with `XGDMatrixFree` once (the Python `QuantileDMatrix` it is
/// handed to does), and what the build did.
#[derive(Debug, Clone)]
pub struct XgbBuilt {
    pub handle: usize,
    pub rows: u64,
    /// Passes over the source, the sample's included.
    pub passes: usize,
    /// Batches handed to XGBoost, over all its passes.
    pub batches: usize,
    /// Seconds drawing the sample and placing the cuts (0 without).
    pub sample_s: f64,
    /// Seconds in XGBoost's construction.
    pub build_s: f64,
    /// Seconds XGBoost or the sampler waited for a frame.
    pub waited_s: f64,
}

/// A matrix handle freed when dropped, unless released.
struct Owned<'a> {
    lib: &'a XgbLib,
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
            // SAFETY: a handle XGBoost made, freed once.
            unsafe { (self.lib.free)(self.h) };
        }
    }
}

fn cstr(s: &str) -> Result<CString, GbdtError> {
    CString::new(s).map_err(|_| GbdtError::msg(format!("`{s}` holds a NUL character")))
}

/// One `__array_interface__`, as JSON, onto `out`.
fn interface(out: &mut String, addr: usize, typestr: &str, shape: &[usize], mask: Option<usize>) {
    let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
    let dims = dims.join(", ");
    let _ = write!(
        out,
        "{{\"data\": [{addr}, true], \"typestr\": \"{typestr}\", \"shape\": [{dims}], \
         \"strides\": null, \"version\": 3, \"mask\": "
    );
    match mask {
        Some(m) => {
            let _ = write!(
                out,
                "{{\"data\": [{m}, true], \"typestr\": \"<t1\", \"shape\": [{dims}], \
                 \"strides\": null, \"version\": 3}}}}"
            );
        }
        None => out.push_str("null}"),
    }
}

fn set_str_info(lib: &XgbLib, h: Handle, field: &str, values: &[CString]) -> Result<(), GbdtError> {
    let ptrs: Vec<*const c_char> = values.iter().map(|s| s.as_ptr()).collect();
    let f = cstr(field)?;
    // SAFETY: a live handle; the strings outlive the call, which copies them.
    let rc = unsafe { (lib.set_str_info)(h, f.as_ptr(), ptrs.as_ptr(), ptrs.len() as u64) };
    lib.check(rc, &format!("setting {field}"))
}

/// The callbacks' state: the reader, the segments of the frame in hand, the
/// one XGBoost is reading, and the first error, which ends the pass.
struct It<'a> {
    lib: &'a XgbLib,
    reader: Reader,
    layout: &'a Layout,
    proxy: Handle,
    pending: VecDeque<Segment>,
    held: Option<Segment>,
    names: &'a [CString],
    types: &'a [CString],
    error: Option<GbdtError>,
    row: u64,
    first_pass_rows: Option<u64>,
    batches: usize,
    json: String,
}

impl It<'_> {
    fn advance(&mut self) -> Result<bool, GbdtError> {
        // XGBoost has read the last batch by the time it asks for the next.
        self.held = None;
        let seg = loop {
            if let Some(s) = self.pending.pop_front() {
                break s;
            }
            match self.reader.next()? {
                Some(df) => self.pending.extend(segments_of(df, self.layout)?),
                None => {
                    self.end_pass()?;
                    return Ok(false);
                }
            }
        };
        let p = self.layout.names.len();
        let n = seg.n;
        let label = checked_f64(&seg, p, self.layout, self.row)?;
        let weight = match self.layout.weight {
            Some(_) => Some(checked_f64(&seg, p + 1, self.layout, self.row)?),
            None => None,
        };
        self.json.clear();
        self.json.push('[');
        for j in 0..p {
            if j > 0 {
                self.json.push_str(", ");
            }
            let num = seg.num(j);
            let mask = num.validity().map(|b| b.as_slice().0.as_ptr() as usize);
            interface(&mut self.json, num.values_addr(), num.typestr(), &[n], mask);
        }
        self.json.push(']');
        let lib = self.lib;
        let data = cstr(&self.json)?;
        // SAFETY: the proxy is live; the buffers the JSON points into stay
        // alive in `self.held` until XGBoost asks for the next batch.
        lib.check(
            unsafe { (lib.set_columnar)(self.proxy, data.as_ptr()) },
            "handing over a batch",
        )?;
        let info = |field: &str, vals: &[f64]| -> Result<(), GbdtError> {
            let mut s = String::new();
            interface(&mut s, vals.as_ptr() as usize, "<f8", &[n], None);
            let (f, s) = (cstr(field)?, cstr(&s)?);
            // SAFETY: as above; XGBoost copies the meta information.
            lib.check(
                unsafe { (lib.set_info)(self.proxy, f.as_ptr(), s.as_ptr()) },
                &format!("setting the {field}"),
            )
        };
        info("label", label)?;
        if let Some(w) = weight {
            info("weight", w)?;
        }
        set_str_info(lib, self.proxy, "feature_name", self.names)?;
        set_str_info(lib, self.proxy, "feature_type", self.types)?;
        self.row += n as u64;
        self.batches += 1;
        self.held = Some(seg);
        Ok(true)
    }

    fn end_pass(&mut self) -> Result<(), GbdtError> {
        match self.first_pass_rows {
            None if self.row == 0 => Err(GbdtError::msg("the source has no rows")),
            None => {
                self.first_pass_rows = Some(self.row);
                Ok(())
            }
            Some(first) if first != self.row => Err(GbdtError::msg(format!(
                "the source gave {first} rows in one pass and {} in another: XGBoost reads it \
                 more than once, and every pass must give the same rows",
                self.row
            ))),
            Some(_) => Ok(()),
        }
    }
}

extern "C" fn xgb_next(h: Handle) -> c_int {
    // SAFETY: `h` is the `It` handed to XGBoost, alive for the whole
    // construction, and XGBoost calls back on the constructing thread.
    let it = unsafe { &mut *(h as *mut It<'_>) };
    if it.error.is_some() {
        return 0;
    }
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| it.advance()));
    match r {
        Ok(Ok(more)) => c_int::from(more),
        Ok(Err(e)) => {
            it.error = Some(e);
            0
        }
        Err(p) => {
            it.error = Some(GbdtError::msg(format!(
                "handing XGBoost a batch panicked: {}",
                panic_message(&p)
            )));
            0
        }
    }
}

extern "C" fn xgb_reset(h: Handle) {
    // SAFETY: as in `xgb_next`.
    let it = unsafe { &mut *(h as *mut It<'_>) };
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        it.pending.clear();
        it.held = None;
        it.reader.restart();
        it.row = 0;
    }));
}

/// The `ref` matrix of the sample's cuts (a plain `DMatrix` of the planted
/// values, which XGBoost sketches exactly).
#[allow(clippy::too_many_arguments)]
fn sample_ref<'a>(
    lib: &'a XgbLib,
    reader: &mut Reader,
    layout: &Layout,
    k: usize,
    opts: &XgbOptions,
    pool: &rayon::ThreadPool,
    names: &[CString],
    types: &[CString],
) -> Result<Owned<'a>, GbdtError> {
    let mut s = Sampler::new(layout.kinds.clone(), k, opts.seed, layout.weight.is_some());
    while let Some(frame) = reader.next()? {
        s.push_any(frame, layout, pool)?;
    }
    if s.rows_seen() == 0 {
        return Err(GbdtError::msg("the source has no rows"));
    }
    let (m, rows) = s.ref_matrix(opts.max_bin, pool);
    dense(
        lib,
        &m,
        rows,
        layout.names.len(),
        opts.threads,
        names,
        types,
    )
}

/// A `DMatrix` of the row-major float32 `values`, `rows` by `p`.
fn dense<'a>(
    lib: &'a XgbLib,
    values: &[f32],
    rows: usize,
    p: usize,
    threads: usize,
    names: &[CString],
    types: &[CString],
) -> Result<Owned<'a>, GbdtError> {
    let mut iface = String::new();
    interface(
        &mut iface,
        values.as_ptr() as usize,
        "<f4",
        &[rows, p],
        None,
    );
    let iface = cstr(&iface)?;
    let cfg = cstr(&format!(
        "{{\"missing\": NaN, \"nthread\": {}}}",
        nthread(threads)
    ))?;
    let mut h: Handle = null_mut();
    // SAFETY: `values` outlives the call, which copies it.
    let rc = unsafe { (lib.from_dense)(iface.as_ptr(), cfg.as_ptr(), &mut h) };
    lib.check(rc, "building the ref matrix of the sample's cuts")?;
    let owned = Owned { lib, h };
    set_str_info(lib, owned.h, "feature_name", names)?;
    set_str_info(lib, owned.h, "feature_type", types)?;
    Ok(owned)
}

fn nthread(threads: usize) -> i64 {
    if threads == 0 { -1 } else { threads as i64 }
}

/// A `QuantileDMatrix` of the frames, binned on `ref_handle`'s cuts when
/// given (a validation set on its training set's), on cuts from a sample of
/// `opts.sample_rows` rows, or on XGBoost's own sketch.
pub fn xgboost_matrix(
    lib: &XgbLib,
    frames: Box<dyn Frames>,
    layout: &Layout,
    ref_handle: Option<usize>,
    opts: &XgbOptions,
) -> Result<XgbBuilt, GbdtError> {
    layout.check()?;
    if opts.max_bin < 2 {
        return Err(GbdtError::msg(format!(
            "max_bin = {}: XGBoost needs 2 bins at least",
            opts.max_bin
        )));
    }
    let pool = conversion_pool(opts.threads)?;
    let names = layout
        .names
        .iter()
        .map(|n| cstr(n))
        .collect::<Result<Vec<_>, _>>()?;
    let types = layout
        .kinds
        .iter()
        .map(|k| cstr(k.xgb_type()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut reader = Reader::new(frames, opts.read_ahead);

    let t = Instant::now();
    let owned_ref = match (ref_handle, opts.sample_rows) {
        (None, Some(k)) => Some(sample_ref(
            lib,
            &mut reader,
            layout,
            k,
            opts,
            &pool,
            &names,
            &types,
        )?),
        _ => None,
    };
    let sample_s = if owned_ref.is_some() {
        t.elapsed().as_secs_f64()
    } else {
        0.0
    };
    reader.restart();
    let refh: Handle = match (ref_handle, &owned_ref) {
        (Some(h), _) => h as Handle,
        (None, Some(o)) => o.h,
        (None, None) => null_mut(),
    };

    let mut proxy: Handle = null_mut();
    // SAFETY: a fresh handle for XGBoost to fill.
    lib.check(
        unsafe { (lib.proxy_create)(&mut proxy) },
        "creating the proxy matrix",
    )?;
    let proxy = Owned { lib, h: proxy };
    let mut it = It {
        lib,
        reader,
        layout,
        proxy: proxy.h,
        pending: VecDeque::new(),
        held: None,
        names: &names,
        types: &types,
        error: None,
        row: 0,
        first_pass_rows: None,
        batches: 0,
        json: String::new(),
    };
    let cfg = cstr(&format!(
        "{{\"missing\": NaN, \"nthread\": {}, \"max_bin\": {}}}",
        nthread(opts.threads),
        opts.max_bin
    ))?;
    let t = Instant::now();
    let mut out: Handle = null_mut();
    // SAFETY: `it` lives on this frame for the whole construction, which
    // calls back only on this thread; `refh` is null or a live matrix.
    let rc = unsafe {
        (lib.from_callback)(
            (&raw mut it).cast(),
            proxy.h,
            refh,
            xgb_reset,
            xgb_next,
            cfg.as_ptr(),
            &mut out,
        )
    };
    let build_s = t.elapsed().as_secs_f64();
    let out = Owned { lib, h: out };
    if let Some(e) = it.error.take() {
        return Err(e);
    }
    lib.check(rc, "building the QuantileDMatrix")?;
    let mut rows = 0u64;
    // SAFETY: a live matrix.
    lib.check(
        unsafe { (lib.num_row)(out.h, &mut rows) },
        "counting the rows",
    )?;
    let passes = it.reader.passes;
    let waited_s = it.reader.waited.as_secs_f64();
    let batches = it.batches;
    drop(it);
    drop(proxy);
    drop(owned_ref);
    Ok(XgbBuilt {
        handle: out.release() as usize,
        rows,
        passes,
        batches,
        sample_s,
        build_s,
        waited_s,
    })
}
