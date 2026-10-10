//! Where a build's frames come from, read ahead of the library on a thread
//! of their own; and each frame cut into segments whose every column is one
//! Arrow array, which is what a library's batch interface takes.
//!
//! A polars frame arrives in as many chunks as the engine made it in
//! (`collect_batches` hands over three per 262,144 rows here); the reader
//! thread joins each column's chunks into one (`rechunk_mut_par`) while the
//! library works on the frame before, since a library's per-batch cost is
//! paid per segment: on 10 million rows of 50 features XGBoost took 4.4 s in
//! 378 segments and 3.9 s in 117 (task 242). Columns read from Arrow IPC
//! files ([`super::IpcFrames`]) arrive one array per record batch.

use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;

use super::{GbdtError, Layout};

/// One frame of a source: a polars frame, or the columns of an Arrow record
/// batch, in the [`Layout`]'s order.
pub enum Frame {
    Df(DataFrame),
    Cols { n: usize, cols: Vec<Box<dyn Array>> },
}

/// A source of frames that can be read more than once: XGBoost reads its
/// input three or four times, and LightGBM twice.
pub trait Frames: Send {
    /// One pass over the source, in order: each frame to `out`, which
    /// returns `false` once it wants no more (the pass then stops early).
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError>;
}

/// Frames already in memory, handed out in order on every pass.
pub struct VecFrames(pub Vec<DataFrame>);

impl Frames for VecFrames {
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError> {
        for df in &self.0 {
            if !out(Frame::Df(df.clone())) {
                break;
            }
        }
        Ok(())
    }
}

/// A polars plan, run on the streaming engine once per pass and handed over
/// in chunks of `chunk_size` rows (`sink_batches`), as the runner reads one.
pub struct PlanFrames {
    plan: LazyFrame,
    chunk_size: NonZeroUsize,
}

impl PlanFrames {
    pub fn new(plan: LazyFrame, chunk_size: NonZeroUsize) -> Self {
        Self { plan, chunk_size }
    }
}

impl Frames for PlanFrames {
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError> {
        let (tx, rx) = sync_channel::<DataFrame>(1);
        let plan = self.plan.clone();
        let chunk = self.chunk_size;
        let engine = std::thread::spawn(move || {
            // A closed channel is the pass giving up; `true` stops the engine.
            let cb = PlanCallback::new(move |df: DataFrame| Ok(tx.send(df).is_err()));
            plan.sink_batches(cb, true, Some(chunk))
                .and_then(|lf| lf.collect_with_engine(Engine::Streaming))
                .map(|_| ())
        });
        let mut wanted = true;
        while let Ok(df) = rx.recv() {
            if !out(Frame::Df(df)) {
                wanted = false;
                break;
            }
        }
        drop(rx);
        let done = engine
            .join()
            .map_err(|_| GbdtError::msg("the polars engine reading the source panicked"))?;
        if wanted {
            done?;
        }
        Ok(())
    }
}

enum Msg {
    Frame(Frame),
    End,
    Fail(GbdtError),
}

/// Frames read ahead of the library: a thread runs each pass of the source
/// and keeps up to `depth` frames waiting. A new pass begins at the first
/// [`Reader::next`] after [`Reader::restart`]; a pass abandoned part way is
/// stopped at its next frame, when the thread finds the channel closed.
pub(crate) struct Reader {
    frames: Arc<Mutex<Box<dyn Frames>>>,
    rx: Option<Receiver<Msg>>,
    ended: bool,
    depth: usize,
    /// Passes begun.
    pub(crate) passes: usize,
    /// Time the library spent waiting for a frame.
    pub(crate) waited: Duration,
}

impl Reader {
    pub(crate) fn new(frames: Box<dyn Frames>, depth: usize) -> Self {
        Self {
            frames: Arc::new(Mutex::new(frames)),
            rx: None,
            ended: false,
            depth: depth.max(1),
            passes: 0,
            waited: Duration::ZERO,
        }
    }

    /// The next pass starts from the first frame.
    pub(crate) fn restart(&mut self) {
        self.rx = None;
        self.ended = false;
    }

    fn start(&mut self) {
        let (tx, rx) = sync_channel(self.depth);
        let frames = Arc::clone(&self.frames);
        std::thread::Builder::new()
            .name("gbdt-reader".into())
            .spawn(move || {
                // One pass at a time: a new pass waits for the last to stop.
                let mut frames = frames.lock().unwrap_or_else(PoisonError::into_inner);
                let mut stopped = false;
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    frames.pass(&mut |frame| {
                        let frame = match frame {
                            Frame::Df(mut df) => {
                                df.rechunk_mut_par();
                                Frame::Df(df)
                            }
                            cols => cols,
                        };
                        stopped = tx.send(Msg::Frame(frame)).is_err();
                        !stopped
                    })
                }));
                if stopped {
                    return;
                }
                let end = match result {
                    Ok(Ok(())) => Msg::End,
                    Ok(Err(e)) => Msg::Fail(e),
                    Err(p) => Msg::Fail(GbdtError::msg(format!(
                        "reading the source panicked: {}",
                        panic_message(&p)
                    ))),
                };
                let _ = tx.send(end);
            })
            .expect("spawning the reader thread");
        self.rx = Some(rx);
        self.passes += 1;
    }

    /// The pass's next frame, or `None` at its end.
    pub(crate) fn next(&mut self) -> Result<Option<Frame>, GbdtError> {
        if self.ended {
            return Ok(None);
        }
        if self.rx.is_none() {
            self.start();
        }
        let t = Instant::now();
        let msg = self.rx.as_ref().expect("started").recv();
        self.waited += t.elapsed();
        match msg {
            Ok(Msg::Frame(df)) => Ok(Some(df)),
            Ok(Msg::End) => {
                self.ended = true;
                Ok(None)
            }
            Ok(Msg::Fail(e)) => {
                self.ended = true;
                Err(e)
            }
            Err(_) => Err(GbdtError::msg(
                "the thread reading the source stopped before the end of its pass",
            )),
        }
    }
}

pub(crate) fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic without a message".into()
    }
}

/// Rows of a frame whose every column is one Arrow array, its validity
/// bitmap (if any null remains) starting at bit 0.
pub(crate) struct Segment {
    pub(crate) n: usize,
    pub(crate) cols: Vec<Box<dyn Array>>,
}

impl Segment {
    pub(crate) fn num(&self, j: usize) -> Num<'_> {
        Num::of(self.cols[j].as_ref()).expect("checked in segments()")
    }
}

/// A frame's segments: a polars frame's from [`segments`], a record
/// batch's columns as one, each checked as [`segments`] checks them.
pub(crate) fn segments_of(frame: Frame, layout: &Layout) -> Result<Vec<Segment>, GbdtError> {
    match frame {
        Frame::Df(df) => segments(&df, layout),
        Frame::Cols { n, cols } => {
            check_kinds(
                cols.iter()
                    .map(|c| (c.as_ref(), format!("{:?}", c.dtype()))),
                layout,
            )?;
            let cols = cols.into_iter().map(aligned).collect();
            Ok(if n == 0 {
                Vec::new()
            } else {
                vec![Segment { n, cols }]
            })
        }
    }
}

/// Each column a primitive numeric one, the label and weight `Float64`.
fn check_kinds<'a>(
    cols: impl Iterator<Item = (&'a dyn Array, String)>,
    layout: &Layout,
) -> Result<(), GbdtError> {
    let p = layout.names.len();
    let mut width = 0;
    for (j, (c, dtype)) in cols.enumerate() {
        width += 1;
        let num = Num::of(c);
        let ok = if j < p {
            num.is_some()
        } else {
            matches!(num, Some(Num::F64(_)))
        };
        if !ok {
            return Err(GbdtError::msg(format!(
                "column `{}` arrived as {dtype}, which is not a numeric column the libraries read",
                column_name(layout, j)
            )));
        }
    }
    if width != layout.width() {
        return Err(GbdtError::msg(format!(
            "a frame of {width} columns where {} were expected",
            layout.width()
        )));
    }
    Ok(())
}

/// `df` cut where any column's chunks end, so that each segment's columns
/// are slices of one chunk apiece: no column is copied. Each column must be
/// a primitive numeric one, and the label and weight `Float64`.
pub(crate) fn segments(df: &DataFrame, layout: &Layout) -> Result<Vec<Segment>, GbdtError> {
    let width = layout.width();
    if df.width() != width {
        return Err(GbdtError::msg(format!(
            "a frame of {} columns where {width} were expected (the features, the label{})",
            df.width(),
            if layout.weight.is_some() {
                ", the weight"
            } else {
                ""
            }
        )));
    }
    let n = df.height();
    let series: Vec<&Series> = df
        .columns()
        .iter()
        .map(|c| c.as_materialized_series())
        .collect();
    let p = layout.names.len();
    for (j, s) in series.iter().enumerate() {
        let name = column_name(layout, j);
        let ok = s.chunks().iter().all(|c| {
            let num = Num::of(c.as_ref());
            if j < p {
                num.is_some()
            } else {
                matches!(num, Some(Num::F64(_)))
            }
        });
        if !ok {
            return Err(GbdtError::msg(format!(
                "column `{name}` arrived as {}, which is not a numeric column the libraries read",
                s.dtype()
            )));
        }
    }
    let mut cuts: Vec<usize> = vec![0, n];
    for s in &series {
        let mut at = 0;
        for c in s.chunks() {
            at += c.len();
            cuts.push(at);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::with_capacity(cuts.len());
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b == a || b > n {
            continue;
        }
        let cols = series.iter().map(|s| slice_of(s, a, b - a)).collect();
        out.push(Segment { n: b - a, cols });
    }
    Ok(out)
}

pub(crate) fn column_name(layout: &Layout, j: usize) -> &str {
    let p = layout.names.len();
    if j < p {
        &layout.names[j]
    } else if j == p {
        &layout.label
    } else {
        layout.weight.as_deref().unwrap_or("weight")
    }
}

fn slice_of(s: &Series, offset: usize, len: usize) -> Box<dyn Array> {
    let mut start = 0;
    for c in s.chunks() {
        if offset < start + c.len() {
            return aligned(c.sliced(offset - start, len));
        }
        start += c.len();
    }
    unreachable!("a segment inside the frame")
}

/// The array with its validity starting at bit 0 (a copy of the bitmap when
/// a slice starts mid-byte), and no validity at all when nothing is null.
pub(crate) fn aligned(a: Box<dyn Array>) -> Box<dyn Array> {
    match a.validity() {
        Some(v) if v.unset_bits() == 0 => a.with_validity(None),
        Some(v) if v.as_slice().1 != 0 => {
            let b: Bitmap = v.iter().collect();
            a.with_validity(Some(b))
        }
        _ => a,
    }
}

#[cfg(target_endian = "little")]
macro_rules! endian {
    ($s:literal) => {
        concat!("<", $s)
    };
}
#[cfg(target_endian = "big")]
macro_rules! endian {
    ($s:literal) => {
        concat!(">", $s)
    };
}

macro_rules! nums {
    ($($v:ident => $t:ty, $ts:expr;)*) => {
        /// A primitive numeric Arrow array, typed.
        pub(crate) enum Num<'a> {
            $($v(&'a PrimitiveArray<$t>),)*
        }

        impl<'a> Num<'a> {
            pub(crate) fn of(a: &'a dyn Array) -> Option<Self> {
                let any = a.as_any();
                $(if let Some(x) = any.downcast_ref::<PrimitiveArray<$t>>() {
                    return Some(Num::$v(x));
                })*
                None
            }

            /// The `__array_interface__` type string of the values.
            pub(crate) fn typestr(&self) -> &'static str {
                match self {
                    $(Num::$v(_) => $ts,)*
                }
            }

            /// The address of the first value.
            pub(crate) fn values_addr(&self) -> usize {
                match self {
                    $(Num::$v(x) => x.values().as_ptr() as usize,)*
                }
            }

            pub(crate) fn validity(&self) -> Option<&'a Bitmap> {
                match self {
                    $(Num::$v(x) => x.validity(),)*
                }
            }

            /// Value `i` as float32, as XGBoost reads every value and as
            /// LightGBM's rows carry it; NaN for a null.
            pub(crate) fn f32_at(&self, i: usize) -> f32 {
                match self {
                    $(Num::$v(x) => {
                        if x.is_valid(i) {
                            x.value(i) as f32
                        } else {
                            f32::NAN
                        }
                    })*
                }
            }

            /// Column `j` of the row-major block `dst` (rows of `stride`
            /// values), from rows `r0..` of this array; NaN for a null.
            pub(crate) fn fill_f32(&self, dst: &mut [f32], stride: usize, j: usize, r0: usize) {
                let rn = dst.len() / stride;
                match self {
                    $(Num::$v(x) => {
                        let vals = &x.values()[r0..r0 + rn];
                        match x.validity() {
                            None => {
                                for (r, &v) in vals.iter().enumerate() {
                                    dst[r * stride + j] = v as f32;
                                }
                            }
                            Some(b) => {
                                for (r, &v) in vals.iter().enumerate() {
                                    dst[r * stride + j] =
                                        if b.get_bit(r0 + r) { v as f32 } else { f32::NAN };
                                }
                            }
                        }
                    })*
                }
            }

            /// The array converted to float64, its validity kept.
            pub(crate) fn to_f64(&self) -> PrimitiveArray<f64> {
                match self {
                    $(Num::$v(x) => {
                        let vals: Vec<f64> = x.values().iter().map(|&v| v as f64).collect();
                        PrimitiveArray::from_vec(vals).with_validity(x.validity().cloned())
                    })*
                }
            }

            /// The array converted to float32, its validity kept.
            pub(crate) fn to_f32(&self) -> PrimitiveArray<f32> {
                match self {
                    $(Num::$v(x) => {
                        let vals: Vec<f32> = x.values().iter().map(|&v| v as f32).collect();
                        PrimitiveArray::from_vec(vals).with_validity(x.validity().cloned())
                    })*
                }
            }
        }
    };
}

nums! {
    F32 => f32, endian!("f4");
    F64 => f64, endian!("f8");
    I8 => i8, "|i1";
    I16 => i16, endian!("i2");
    I32 => i32, endian!("i4");
    I64 => i64, endian!("i8");
    U8 => u8, "|u1";
    U16 => u16, endian!("u2");
    U32 => u32, endian!("u4");
    U64 => u64, endian!("u8");
}

impl<'a> Num<'a> {
    /// The values of a `Float64` column (the label, the weight).
    pub(crate) fn f64_values(&self) -> Option<&'a [f64]> {
        match self {
            Num::F64(x) => Some(x.values().as_slice()),
            _ => None,
        }
    }
}

/// The label's or the weight's values, refused by name when one is null or
/// not finite, or (for a weight) negative.
pub(crate) fn checked_f64<'a>(
    seg: &'a Segment,
    j: usize,
    layout: &Layout,
    row0: u64,
) -> Result<&'a [f64], GbdtError> {
    let num = seg.num(j);
    let name = column_name(layout, j);
    let what = if j == layout.names.len() {
        "label"
    } else {
        "weight"
    };
    if let Some(v) = num.validity() {
        let at = v.iter().position(|ok| !ok).unwrap_or(0) as u64;
        return Err(GbdtError::msg(format!(
            "the {what} column `{name}` has a null at row {}: every row needs a {what}",
            row0 + at
        )));
    }
    let vals = num.f64_values().expect("checked in segments()");
    let bad = if what == "label" {
        vals.iter().position(|v| !v.is_finite())
    } else {
        vals.iter().position(|v| !v.is_finite() || *v < 0.0)
    };
    if let Some(i) = bad {
        return Err(GbdtError::msg(format!(
            "the {what} column `{name}` holds {:?} at row {}: a {what} must be {}",
            vals[i],
            row0 + i as u64,
            if what == "label" {
                "finite"
            } else {
                "finite and not negative"
            }
        )));
    }
    Ok(vals)
}
