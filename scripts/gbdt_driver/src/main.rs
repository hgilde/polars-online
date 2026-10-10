//! The measurement driver behind options C and D of docs/GBDT-ADAPTER.md
//! (PLAN task 235): Arrow IPC files read in Rust, cuts computed in Rust, and
//! XGBoost's and LightGBM's C APIs driven from Rust, with the shared
//! libraries their Python wheels ship loaded at run time (`dlopen`; nothing
//! is linked). A research prototype: no part of the package.
//!
//! ```text
//! gbdt-driver sketch DIR --out cuts.txt [--sketch digest|sample] [--compression 1000]
//! gbdt-driver xgb DIR --lib libxgboost.dylib --model out.json --feed dense|columnar|codes
//!             [--cuts cuts.txt] [--rounds 50] [--threads 8]
//! gbdt-driver lgb DIR --lib lib_lightgbm.dylib --model out.txt --rows N --feed f64|f32|codes
//!             [--cuts cuts.txt] [--forced] [--rounds 50] [--threads 8]
//! ```
//!
//! DIR holds `part-*.arrow` (uncompressed Arrow IPC, as `scripts/gbdt_adapter.py
//! to-ipc` writes them): Float64 columns `x0..`, a dictionary column `cat` and
//! a Float64 target `y`. Prints one JSON line of timings and peaks.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt8Type, UInt16Type, UInt32Type};
use arrow_array::{Array, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_ipc::reader::FileReader;
use arrow_schema::DataType;
use libloading::Library;
use rayon::prelude::*;

/// Rows a conversion task writes at once: 4,096 rows of 51 Float32 columns
/// is 0.8 MB, inside a core's cache.
const BLOCK: usize = 4096;
/// The code a missing value takes when batches are fed quantized.
const MISSING_CODE: u8 = 255;
/// Edges a quantized feature may have: 254 edges make 255 bins, codes 0..=254.
const MAX_EDGES: usize = 254;

// ---------------------------------------------------------------- arguments

struct Args {
    cmd: String,
    dir: PathBuf,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().expect("command");
        let dir = PathBuf::from(it.next().expect("data directory"));
        let (mut kv, mut flags) = (HashMap::new(), Vec::new());
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let k = rest[i].trim_start_matches("--").to_string();
            if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                kv.insert(k, rest[i + 1].clone());
                i += 2;
            } else {
                flags.push(k);
                i += 1;
            }
        }
        Self {
            cmd,
            dir,
            kv,
            flags,
        }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(String::as_str)
    }
    fn num(&self, k: &str, default: usize) -> usize {
        self.get(k)
            .map_or(default, |v| v.parse().expect("a number"))
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

// ---------------------------------------------------------- time and memory

/// macOS's physical footprint now and its lifetime peak (`proc_pid_rusage`).
fn footprint() -> (u64, u64) {
    // SAFETY: `info` is a plain struct the call fills; flavor 4 is its version.
    unsafe {
        let mut info: libc::rusage_info_v4 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        );
        assert_eq!(rc, 0, "proc_pid_rusage");
        (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
    }
}

static PHASE_PEAK: AtomicU64 = AtomicU64::new(0);
static STOP: AtomicBool = AtomicBool::new(false);

/// Wall time and peak footprint per phase, sampled every 20 ms.
struct Phases {
    t: Instant,
    name: String,
    done: Vec<(String, f64, f64)>,
}

impl Phases {
    fn start(name: &str) -> Self {
        PHASE_PEAK.store(footprint().0, Ordering::Relaxed);
        std::thread::spawn(|| {
            while !STOP.load(Ordering::Relaxed) {
                PHASE_PEAK.fetch_max(footprint().0, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        });
        Self {
            t: Instant::now(),
            name: name.into(),
            done: Vec::new(),
        }
    }
    fn next(&mut self, name: &str) {
        let now = footprint().0;
        let peak = PHASE_PEAK.swap(now, Ordering::Relaxed).max(now);
        self.done.push((
            self.name.clone(),
            self.t.elapsed().as_secs_f64(),
            peak as f64 / 1e9,
        ));
        self.t = Instant::now();
        self.name = name.into();
    }
    fn json(mut self) -> String {
        self.next("end");
        STOP.store(true, Ordering::Relaxed);
        let parts: Vec<String> = self
            .done
            .iter()
            .map(|(n, s, g)| format!("\"{n}\": {{\"s\": {s:.2}, \"peak_gb\": {g:.2}}}"))
            .collect();
        format!(
            "\"phases\": {{{}}}, \"peak_gb\": {:.2}",
            parts.join(", "),
            footprint().1 as f64 / 1e9
        )
    }
}

// ------------------------------------------------------------------ source

/// Batches a reader thread may decode ahead of the library, as Polars'
/// `collect_batches` reads ahead of its consumer.
const READ_AHEAD: usize = 4;

/// The Arrow IPC parts of a directory, read batch by batch on a thread of
/// their own, restartable: a library's every further pass reads the files
/// again. `read_s` is the time the library waited for a batch.
struct Source {
    files: Vec<PathBuf>,
    rx: Option<std::sync::mpsc::Receiver<RecordBatch>>,
    done: bool,
    read_any: bool,
    read_s: f64,
    passes: usize,
    pass_ends: Vec<f64>,
    t0: Instant,
}

impl Source {
    fn new(dir: &Path) -> Self {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("data directory")
            .map(|e| e.unwrap().path())
            .filter(|p| {
                let n = p.file_name().unwrap().to_string_lossy();
                n.starts_with("part-") && n.ends_with(".arrow")
            })
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no part-*.arrow in {}", dir.display());
        Self {
            files,
            rx: None,
            done: false,
            read_any: false,
            read_s: 0.0,
            passes: 0,
            pass_ends: Vec::new(),
            t0: Instant::now(),
        }
    }
    fn reset(&mut self) {
        if self.read_any {
            self.pass_ends.push(self.t0.elapsed().as_secs_f64());
        }
        // Dropping the receiver ends a reader thread still running.
        self.rx = None;
        self.done = false;
        self.read_any = false;
    }
    fn next(&mut self) -> Option<RecordBatch> {
        if self.done {
            return None;
        }
        let t = Instant::now();
        if self.rx.is_none() {
            let (tx, rx) = std::sync::mpsc::sync_channel(READ_AHEAD);
            let files = self.files.clone();
            std::thread::spawn(move || {
                for f in files {
                    let f = File::open(&f).expect("open part");
                    for b in FileReader::try_new(BufReader::new(f), None).expect("ipc") {
                        if tx.send(b.expect("record batch")).is_err() {
                            return;
                        }
                    }
                }
            });
            self.rx = Some(rx);
            self.passes += 1;
        }
        let out = self.rx.as_ref().unwrap().recv().ok();
        self.read_s += t.elapsed().as_secs_f64();
        match out {
            Some(_) => self.read_any = true,
            None => self.done = true,
        }
        out
    }
}

/// One batch's columns, borrowed from its Arrow buffers.
struct View<'a> {
    n: usize,
    x: Vec<(&'a [f64], Option<&'a NullBuffer>)>,
    cat: Vec<u8>,
    y: &'a [f64],
}

fn view(b: &RecordBatch, nx: usize) -> View<'_> {
    let x = (0..nx)
        .map(|j| {
            let a = b
                .column_by_name(&format!("x{j}"))
                .expect("feature")
                .as_primitive::<Float64Type>();
            assert_eq!(a.offset(), 0, "a sliced column");
            (a.values().as_ref(), a.nulls())
        })
        .collect();
    let c = b.column_by_name("cat").expect("cat");
    let cat: Vec<u8> = match c.data_type() {
        DataType::Dictionary(k, _) => match k.as_ref() {
            DataType::UInt8 => c.as_dictionary::<UInt8Type>().keys().values().to_vec(),
            DataType::UInt16 => c
                .as_dictionary::<UInt16Type>()
                .keys()
                .values()
                .iter()
                .map(|&k| k as u8)
                .collect(),
            DataType::UInt32 => c
                .as_dictionary::<UInt32Type>()
                .keys()
                .values()
                .iter()
                .map(|&k| k as u8)
                .collect(),
            other => panic!("cat keys {other:?}"),
        },
        DataType::UInt8 => c.as_primitive::<UInt8Type>().values().to_vec(),
        other => panic!("cat {other:?}"),
    };
    let y = b
        .column_by_name("y")
        .expect("y")
        .as_primitive::<Float64Type>()
        .values()
        .as_ref();
    View {
        n: b.num_rows(),
        x,
        cat,
        y,
    }
}

fn n_features(dir: &Path) -> usize {
    let mut s = Source::new(dir);
    let b = s.next().expect("a batch");
    b.schema()
        .fields()
        .iter()
        .filter(|f| f.name().starts_with('x'))
        .count()
}

// ------------------------------------------------------------- conversion

/// Row-major, in parallel over blocks of rows: what `LGBM_DatasetPushRows`
/// takes, and XGBoost's dense input. A null becomes NaN; the category's code
/// is the last column.
fn rows<T: Copy + Send + Sync>(v: &View, cast: fn(f64) -> T, nan: T) -> Vec<T> {
    let p = v.x.len() + 1;
    let mut out = vec![nan; v.n * p];
    out.par_chunks_mut(BLOCK * p)
        .enumerate()
        .for_each(|(bi, chunk)| {
            let r0 = bi * BLOCK;
            let rn = chunk.len() / p;
            for (j, (vals, nulls)) in v.x.iter().enumerate() {
                match nulls {
                    None => {
                        for r in 0..rn {
                            chunk[r * p + j] = cast(vals[r0 + r]);
                        }
                    }
                    Some(nb) => {
                        for r in 0..rn {
                            if nb.is_valid(r0 + r) {
                                chunk[r * p + j] = cast(vals[r0 + r]);
                            }
                        }
                    }
                }
            }
            for r in 0..rn {
                chunk[r * p + p - 1] = cast(f64::from(v.cat[r0 + r]));
            }
        });
    out
}

/// Each feature's bin code, in parallel over features: the number of edges
/// at or below the value, `MISSING_CODE` for a null.
fn codes(v: &View, cuts: &[Vec<f64>]) -> Vec<Vec<u8>> {
    v.x.par_iter()
        .zip(cuts.par_iter())
        .map(|((vals, nulls), edges)| {
            let mut out = vec![MISSING_CODE; v.n];
            for (i, o) in out.iter_mut().enumerate() {
                if nulls.is_none_or(|nb| nb.is_valid(i)) {
                    *o = edges.partition_point(|&e| e <= vals[i]) as u8;
                }
            }
            out
        })
        .collect()
}

/// A `__array_interface__` as XGBoost reads it, encoded as JSON.
fn interface(ptr: usize, typestr: &str, shape: &[usize], mask: Option<usize>) -> String {
    let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
    let m = match mask {
        Some(m) => format!(
            "{{\"data\": [{m}, true], \"typestr\": \"<t1\", \"shape\": [{}], \"strides\": null, \"version\": 3}}",
            shape[0]
        ),
        None => "null".into(),
    };
    format!(
        "{{\"data\": [{ptr}, true], \"typestr\": \"{typestr}\", \"shape\": [{}], \"strides\": null, \"version\": 3, \"mask\": {m}}}",
        shape.join(", ")
    )
}

// ------------------------------------------------------------------ sketch

/// The merging t-digest of `crates/online-core/src/audit.rs` (Dunning &
/// Ertl 2019, uniform rank resolution), copied with its compression made a
/// parameter: that module keeps it private, at 200.
struct Digest {
    c: Vec<(f64, f64)>,
    buf: Vec<f64>,
    comp: f64,
    min: f64,
    max: f64,
}

impl Digest {
    fn new(comp: f64) -> Self {
        Self {
            c: Vec::new(),
            buf: Vec::with_capacity(256),
            comp,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    fn add(&mut self, v: f64) {
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        self.buf.push(v);
        if self.buf.len() >= 256 {
            let mut all = std::mem::take(&mut self.c);
            all.extend(self.buf.drain(..).map(|v| (v, 1.0)));
            self.c = fold(all, self.comp);
        }
    }
    /// `MAX_EDGES` edges at evenly spaced ranks, read from the centroids as
    /// `audit.rs`'s `Cdf::quantile` reads them, duplicates dropped.
    fn edges(&self) -> Vec<f64> {
        let mut all = self.c.clone();
        all.extend(self.buf.iter().map(|&v| (v, 1.0)));
        let c = fold(all, self.comp);
        let n: f64 = c.iter().map(|x| x.1).sum();
        let mut pts = vec![(self.min, 0.0)];
        let mut below = 0.0;
        for &(m, w) in &c {
            pts.push((m, below + w / 2.0));
            below += w;
        }
        pts.push((self.max, n));
        let q = |r: f64| -> f64 {
            let i = pts.partition_point(|p| p.1 < r).clamp(1, pts.len() - 1);
            let (a, b) = (pts[i - 1], pts[i]);
            if b.1 > a.1 {
                a.0 + (b.0 - a.0) * ((r - a.1) / (b.1 - a.1))
            } else {
                b.0
            }
        };
        dedup(
            (1..=MAX_EDGES)
                .map(|k| q(n * k as f64 / (MAX_EDGES + 1) as f64))
                .collect(),
        )
    }
}

fn fold(mut all: Vec<(f64, f64)>, comp: f64) -> Vec<(f64, f64)> {
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = all.iter().map(|c| c.1).sum();
    let cap = (2.0 * total / comp).max(1.0);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (m, w) in all {
        match out.last_mut() {
            Some(cur) if cur.1 + w <= cap => {
                let s = cur.1 + w;
                cur.0 += (m - cur.0) * (w / s);
                cur.1 = s;
            }
            _ => out.push((m, w)),
        }
    }
    out
}

fn dedup(mut e: Vec<f64>) -> Vec<f64> {
    e.retain(|v| v.is_finite());
    e.dedup();
    e
}

/// A row kept in a uniform sample of about `k` rows in `n`: a hash of the
/// row's index against a threshold, the same rows in every column.
fn kept(i: u64, k: usize, n: usize) -> bool {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z as f64) < (k as f64 / n as f64) * (u64::MAX as f64)
}

fn sample_edges(mut vals: Vec<f64>) -> Vec<f64> {
    vals.sort_by(f64::total_cmp);
    let m = vals.len();
    dedup(
        (1..=MAX_EDGES)
            .map(|k| vals[(k * m / (MAX_EDGES + 1)).min(m - 1)])
            .collect(),
    )
}

/// One pass over the files: each feature's edges, from the digest or from a
/// uniform sample of 200,000 rows, in parallel over features.
fn sketch(a: &Args) -> String {
    let nx = n_features(&a.dir);
    let how = a.get("sketch").unwrap_or("digest").to_string();
    let comp = a.num("compression", 1000) as f64;
    let rows = a.num("rows", 0);
    let mut ph = Phases::start("sketch");
    let mut src = Source::new(&a.dir);
    let mut digests: Vec<Digest> = (0..nx).map(|_| Digest::new(comp)).collect();
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); nx];
    let mut start = 0u64;
    while let Some(b) = src.next() {
        let v = view(&b, nx);
        if how == "digest" {
            digests
                .par_iter_mut()
                .zip(v.x.par_iter())
                .for_each(|(d, (vals, nulls))| {
                    for (i, &x) in vals.iter().enumerate() {
                        if nulls.is_none_or(|nb| nb.is_valid(i)) {
                            d.add(x);
                        }
                    }
                });
        } else {
            let keep: Vec<usize> = (0..v.n)
                .filter(|&i| kept(start + i as u64, 200_000, rows))
                .collect();
            samples
                .par_iter_mut()
                .zip(v.x.par_iter())
                .for_each(|(s, (vals, nulls))| {
                    for &i in &keep {
                        if nulls.is_none_or(|nb| nb.is_valid(i)) {
                            s.push(vals[i]);
                        }
                    }
                });
        }
        start += v.n as u64;
    }
    ph.next("edges");
    let edges: Vec<Vec<f64>> = if how == "digest" {
        digests.par_iter().map(Digest::edges).collect()
    } else {
        samples.into_par_iter().map(sample_edges).collect()
    };
    let out = a.get("out").expect("--out");
    let mut f = File::create(out).expect("cuts file");
    for e in &edges {
        let line: Vec<String> = e.iter().map(|v| format!("{v:e}")).collect();
        writeln!(f, "{}", line.join(" ")).unwrap();
    }
    let nedges: usize = edges.iter().map(Vec::len).sum();
    format!(
        "{{\"cmd\": \"sketch\", \"sketch\": \"{how}\", \"compression\": {comp}, \"read_s\": {:.2}, \"edges\": {nedges}, {}}}",
        src.read_s,
        ph.json()
    )
}

fn read_cuts(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .expect("cuts file")
        .lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

// ----------------------------------------------------------------- XGBoost

type Handle = *mut c_void;

/// The XGBoost C functions used, from `include/xgboost/c_api.h` at v3.4.1.
#[allow(clippy::type_complexity)]
struct Xgb {
    _lib: Library,
    last_error: unsafe extern "C" fn() -> *const c_char,
    proxy_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    from_callback: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        extern "C" fn(Handle),
        extern "C" fn(Handle) -> c_int,
        *const c_char,
        *mut Handle,
    ) -> c_int,
    set_dense: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
    set_columnar: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
    set_info: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
    set_str_info: unsafe extern "C" fn(Handle, *const c_char, *const *const c_char, u64) -> c_int,
    booster_str_info:
        unsafe extern "C" fn(Handle, *const c_char, *const *const c_char, u64) -> c_int,
    from_dense: unsafe extern "C" fn(*const c_char, *const c_char, *mut Handle) -> c_int,
    booster_create: unsafe extern "C" fn(*const Handle, u64, *mut Handle) -> c_int,
    set_param: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
    update: unsafe extern "C" fn(Handle, c_int, Handle) -> c_int,
    save: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        // SAFETY: the symbol's type is the one c_api.h declares for it.
        unsafe { *$lib.get($name).unwrap() }
    };
}

impl Xgb {
    fn load(path: &str) -> Self {
        // SAFETY: loading a shared library runs its initializers; these are
        // XGBoost's own, as its Python package loads it.
        let lib = unsafe { Library::new(path) }.expect("libxgboost");
        Self {
            last_error: sym!(lib, b"XGBGetLastError\0"),
            proxy_create: sym!(lib, b"XGProxyDMatrixCreate\0"),
            from_callback: sym!(lib, b"XGQuantileDMatrixCreateFromCallback\0"),
            set_dense: sym!(lib, b"XGProxyDMatrixSetDataDense\0"),
            set_columnar: sym!(lib, b"XGProxyDMatrixSetDataColumnar\0"),
            set_info: sym!(lib, b"XGDMatrixSetInfoFromInterface\0"),
            set_str_info: sym!(lib, b"XGDMatrixSetStrFeatureInfo\0"),
            booster_str_info: sym!(lib, b"XGBoosterSetStrFeatureInfo\0"),
            from_dense: sym!(lib, b"XGDMatrixCreateFromDense\0"),
            booster_create: sym!(lib, b"XGBoosterCreate\0"),
            set_param: sym!(lib, b"XGBoosterSetParam\0"),
            update: sym!(lib, b"XGBoosterUpdateOneIter\0"),
            save: sym!(lib, b"XGBoosterSaveModel\0"),
            _lib: lib,
        }
    }
    fn check(&self, rc: c_int) {
        if rc != 0 {
            // SAFETY: XGBoost returns a NUL-terminated message it owns.
            let msg = unsafe { CStr::from_ptr((self.last_error)()) };
            panic!("xgboost: {}", msg.to_string_lossy());
        }
    }
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[derive(Clone, Copy, PartialEq)]
enum Feed {
    Dense,
    Columnar,
    Codes,
}

struct XgbIter<'a> {
    x: &'a Xgb,
    src: Source,
    proxy: Handle,
    feed: Feed,
    nx: usize,
    cuts: Vec<Vec<f64>>,
    names: Vec<CString>,
    types: Vec<CString>,
    // What the last batch handed XGBoost points into; kept until the next.
    held: (Option<RecordBatch>, Vec<f32>, Vec<Vec<u8>>, Vec<u8>),
    conv_s: f64,
    batches: usize,
}

impl XgbIter<'_> {
    fn str_info(&self, field: &str, v: &[CString]) {
        let ptrs: Vec<*const c_char> = v.iter().map(|s| s.as_ptr()).collect();
        // SAFETY: the proxy is live and the strings outlive the call.
        self.x.check(unsafe {
            (self.x.set_str_info)(
                self.proxy,
                c(field).as_ptr(),
                ptrs.as_ptr(),
                ptrs.len() as u64,
            )
        });
    }
}

extern "C" fn xgb_next(h: Handle) -> c_int {
    // SAFETY: `h` is the `XgbIter` handed to XGBoost, alive for the call.
    let it = unsafe { &mut *(h as *mut XgbIter) };
    let Some(b) = it.src.next() else { return 0 };
    let t = Instant::now();
    let v = view(&b, it.nx);
    let n = v.n;
    let p = it.nx + 1;
    let (data, columnar, dense, cols, cat) = match it.feed {
        Feed::Dense => {
            let d = rows(&v, |x| x as f32, f32::NAN);
            (
                interface(d.as_ptr() as usize, "<f4", &[n, p], None),
                false,
                d,
                Vec::new(),
                Vec::new(),
            )
        }
        Feed::Columnar => {
            let mut parts: Vec<String> =
                v.x.iter()
                    .map(|(vals, nulls)| {
                        let mask = nulls.map(|nb| {
                            assert_eq!(nb.offset(), 0, "a validity bitmap at a bit offset");
                            nb.buffer().as_ptr() as usize
                        });
                        interface(vals.as_ptr() as usize, "<f8", &[n], mask)
                    })
                    .collect();
            let cat = v.cat.clone();
            parts.push(interface(cat.as_ptr() as usize, "|u1", &[n], None));
            (
                format!("[{}]", parts.join(", ")),
                true,
                Vec::new(),
                Vec::new(),
                cat,
            )
        }
        Feed::Codes => {
            let cols = codes(&v, &it.cuts);
            let cat = v.cat.clone();
            let mut parts: Vec<String> = cols
                .iter()
                .map(|c| interface(c.as_ptr() as usize, "|u1", &[n], None))
                .collect();
            parts.push(interface(cat.as_ptr() as usize, "|u1", &[n], None));
            (
                format!("[{}]", parts.join(", ")),
                true,
                Vec::new(),
                cols,
                cat,
            )
        }
    };
    let label = interface(v.y.as_ptr() as usize, "<f8", &[n], None);
    it.conv_s += t.elapsed().as_secs_f64();
    let x = it.x;
    // SAFETY: every pointer in the JSON stays alive in `it.held` until the
    // next call, which is as long as XGBoost reads it.
    unsafe {
        let data = c(&data);
        x.check(if columnar {
            (x.set_columnar)(it.proxy, data.as_ptr())
        } else {
            (x.set_dense)(it.proxy, data.as_ptr())
        });
        x.check((x.set_info)(
            it.proxy,
            c("label").as_ptr(),
            c(&label).as_ptr(),
        ));
    }
    let (names, types) = (it.names.clone(), it.types.clone());
    it.str_info("feature_name", &names);
    it.str_info("feature_type", &types);
    drop(v);
    it.held = (Some(b), dense, cols, cat);
    it.batches += 1;
    1
}

extern "C" fn xgb_reset(h: Handle) {
    // SAFETY: as in `xgb_next`.
    let it = unsafe { &mut *(h as *mut XgbIter) };
    it.src.reset();
}

/// A reference matrix whose cuts are `cuts`: each column holds its feature's
/// edges below one value lower than all of them, which XGBoost takes as the
/// feature's minimum; the category column holds its codes. Padded by
/// repeating a column's last value.
fn xgb_ref(
    x: &Xgb,
    cuts: &[Vec<f64>],
    ncat: usize,
    names: &[CString],
    types: &[CString],
) -> Handle {
    let p = cuts.len() + 1;
    let cols: Vec<Vec<f32>> = cuts
        .iter()
        .map(|e| {
            let lo = e[0] - e[0].abs().max(1.0);
            std::iter::once(lo)
                .chain(e.iter().copied())
                .map(|v| v as f32)
                .collect()
        })
        .chain(std::iter::once((0..ncat).map(|k| k as f32).collect()))
        .collect();
    let m = cols.iter().map(Vec::len).max().unwrap();
    let mut d = vec![0f32; m * p];
    for (j, col) in cols.iter().enumerate() {
        for r in 0..m {
            d[r * p + j] = col[r.min(col.len() - 1)];
        }
    }
    let mut h: Handle = std::ptr::null_mut();
    let cfg = c("{\"missing\": NaN, \"nthread\": 8}");
    // SAFETY: `d` outlives the call, which copies it.
    x.check(unsafe {
        (x.from_dense)(
            c(&interface(d.as_ptr() as usize, "<f4", &[m, p], None)).as_ptr(),
            cfg.as_ptr(),
            &mut h,
        )
    });
    for (field, v) in [("feature_name", names), ("feature_type", types)] {
        let ptrs: Vec<*const c_char> = v.iter().map(|s| s.as_ptr()).collect();
        // SAFETY: as above.
        x.check(unsafe {
            (x.set_str_info)(h, c(field).as_ptr(), ptrs.as_ptr(), ptrs.len() as u64)
        });
    }
    h
}

fn xgb(a: &Args) -> String {
    let nx = n_features(&a.dir);
    let threads = a.num("threads", 8);
    let rounds = a.num("rounds", 50);
    let feed = match a.get("feed").unwrap_or("dense") {
        "dense" => Feed::Dense,
        "columnar" => Feed::Columnar,
        "codes" => Feed::Codes,
        f => panic!("feed {f}"),
    };
    let cuts = a.get("cuts").map(read_cuts).unwrap_or_default();
    let x = Xgb::load(a.get("lib").expect("--lib"));
    let names: Vec<CString> = (0..nx)
        .map(|j| c(&format!("x{j}")))
        .chain(std::iter::once(c("cat")))
        .collect();
    let types: Vec<CString> = (0..nx)
        .map(|_| c("q"))
        .chain(std::iter::once(c("c")))
        .collect();
    let mut ph = Phases::start("bin");
    let mut proxy: Handle = std::ptr::null_mut();
    // SAFETY: a fresh handle for XGBoost to fill.
    x.check(unsafe { (x.proxy_create)(&mut proxy) });
    let refm = if feed == Feed::Codes {
        let identity: Vec<Vec<f64>> = (0..nx)
            .map(|_| (1..=MAX_EDGES).map(|k| k as f64).collect())
            .collect();
        xgb_ref(&x, &identity, 32, &names, &types)
    } else if !cuts.is_empty() {
        xgb_ref(&x, &cuts, 32, &names, &types)
    } else {
        std::ptr::null_mut()
    };
    let mut it = XgbIter {
        x: &x,
        src: Source::new(&a.dir),
        proxy,
        feed,
        nx,
        cuts,
        names,
        types,
        held: (None, Vec::new(), Vec::new(), Vec::new()),
        conv_s: 0.0,
        batches: 0,
    };
    let missing = if feed == Feed::Codes { "255" } else { "NaN" };
    let cfg = c(&format!(
        "{{\"missing\": {missing}, \"nthread\": {threads}, \"max_bin\": 256}}"
    ));
    let mut dtrain: Handle = std::ptr::null_mut();
    // SAFETY: `it` lives on this frame for the whole construction.
    x.check(unsafe {
        (x.from_callback)(
            (&mut it as *mut XgbIter).cast(),
            proxy,
            refm,
            xgb_reset,
            xgb_next,
            cfg.as_ptr(),
            &mut dtrain,
        )
    });
    it.src.reset();
    ph.next("boost");
    let mut bst: Handle = std::ptr::null_mut();
    // SAFETY: handles live until the process ends.
    unsafe {
        x.check((x.booster_create)(&dtrain, 1, &mut bst));
        // The names and types travel to the booster as Python's `Booster`
        // sends them: the dump and a categorical split need them.
        for (field, v) in [("feature_name", &it.names), ("feature_type", &it.types)] {
            let ptrs: Vec<*const c_char> = v.iter().map(|s| s.as_ptr()).collect();
            x.check((x.booster_str_info)(
                bst,
                c(field).as_ptr(),
                ptrs.as_ptr(),
                ptrs.len() as u64,
            ));
        }
        for (k, v) in [
            ("tree_method", "hist".to_string()),
            ("max_depth", "6".into()),
            ("eta", "0.1".into()),
            ("max_bin", "256".into()),
            ("nthread", threads.to_string()),
            ("objective", "reg:squarederror".into()),
            ("seed", "0".into()),
        ] {
            x.check((x.set_param)(bst, c(k).as_ptr(), c(&v).as_ptr()));
        }
        for i in 0..rounds {
            x.check((x.update)(bst, i as c_int, dtrain));
        }
        x.check((x.save)(bst, c(a.get("model").expect("--model")).as_ptr()));
    }
    let ends: Vec<String> = it.src.pass_ends.iter().map(|t| format!("{t:.2}")).collect();
    format!(
        "{{\"cmd\": \"xgb\", \"read_s\": {:.2}, \"convert_s\": {:.2}, \"source_passes\": {}, \"batches\": {}, \"pass_ends_s\": [{}], {}}}",
        it.src.read_s,
        it.conv_s,
        it.src.passes,
        it.batches,
        ends.join(", "),
        ph.json()
    )
}

// ---------------------------------------------------------------- LightGBM

/// The LightGBM C functions used, from `include/LightGBM/c_api.h` at v4.7.0.
#[allow(clippy::type_complexity)]
struct Lgb {
    _lib: Library,
    last_error: unsafe extern "C" fn() -> *const c_char,
    sample_indices: unsafe extern "C" fn(i32, *const c_char, *mut c_void, *mut i32) -> c_int,
    from_sampled: unsafe extern "C" fn(
        *mut *mut f64,
        *mut *mut c_int,
        i32,
        *const c_int,
        i32,
        i32,
        i64,
        *const c_char,
        *mut Handle,
    ) -> c_int,
    push_rows: unsafe extern "C" fn(Handle, *const c_void, c_int, i32, i32, i32) -> c_int,
    set_field: unsafe extern "C" fn(Handle, *const c_char, *const c_void, c_int, c_int) -> c_int,
    booster_create: unsafe extern "C" fn(Handle, *const c_char, *mut Handle) -> c_int,
    update: unsafe extern "C" fn(Handle, *mut c_int) -> c_int,
    save: unsafe extern "C" fn(Handle, c_int, c_int, c_int, *const c_char) -> c_int,
}

impl Lgb {
    fn load(path: &str) -> Self {
        // SAFETY: as for XGBoost: LightGBM's own initializers.
        let lib = unsafe { Library::new(path) }.expect("lib_lightgbm");
        Self {
            last_error: sym!(lib, b"LGBM_GetLastError\0"),
            sample_indices: sym!(lib, b"LGBM_SampleIndices\0"),
            from_sampled: sym!(lib, b"LGBM_DatasetCreateFromSampledColumn\0"),
            push_rows: sym!(lib, b"LGBM_DatasetPushRows\0"),
            set_field: sym!(lib, b"LGBM_DatasetSetField\0"),
            booster_create: sym!(lib, b"LGBM_BoosterCreate\0"),
            update: sym!(lib, b"LGBM_BoosterUpdateOneIter\0"),
            save: sym!(lib, b"LGBM_BoosterSaveModel\0"),
            _lib: lib,
        }
    }
    fn check(&self, rc: c_int) {
        if rc != 0 {
            // SAFETY: LightGBM returns a NUL-terminated message it owns.
            let msg = unsafe { CStr::from_ptr((self.last_error)()) };
            panic!("lightgbm: {}", msg.to_string_lossy());
        }
    }
}

fn lgb(a: &Args) -> String {
    let nx = n_features(&a.dir);
    let p = nx + 1;
    let threads = a.num("threads", 8);
    let rounds = a.num("rounds", 50);
    let total = a.num("rows", 0);
    assert!(total > 0, "--rows: the row count, from the files' metadata");
    let feed = a.get("feed").unwrap_or("f64").to_string();
    let cuts = a.get("cuts").map(read_cuts).unwrap_or_default();
    let mut params = format!(
        "objective=regression num_leaves=63 learning_rate=0.1 max_bin=255 num_threads={threads} seed=0 \
         data_random_seed=1 deterministic=true force_col_wise=true verbose=-1 categorical_feature={nx}"
    );
    if a.flag("forced") {
        let path = format!("{}.forced.json", a.get("model").unwrap());
        let items: Vec<String> = cuts
            .iter()
            .enumerate()
            .map(|(j, e)| {
                // LightGBM adds bounds at -0, +0 and infinity first, then
                // forced bounds while fewer than `max_bin` (255) are placed,
                // in the order given (`FindBinWithPredefinedBin`,
                // src/io/bin.cpp): past 252, the highest would be dropped.
                // So 252 of the edges, evenly spaced.
                let keep = e.len().min(252);
                let e: Vec<f64> = (0..keep)
                    .map(|i| {
                        e[if keep > 1 {
                            i * (e.len() - 1) / (keep - 1)
                        } else {
                            0
                        }]
                    })
                    .collect();
                let b: Vec<String> = e.iter().map(|v| format!("{v:e}")).collect();
                format!(
                    "{{\"feature\": {j}, \"bin_upper_bound\": [{}]}}",
                    b.join(", ")
                )
            })
            .collect();
        std::fs::write(&path, format!("[{}]", items.join(", "))).unwrap();
        params.push_str(&format!(" forcedbins_filename={path}"));
    }
    let l = Lgb::load(a.get("lib").expect("--lib"));
    let cparams = c(&params);
    let mut ph = Phases::start("sample");
    let mut idx = vec![0i32; total.min(200_000)];
    let mut got = 0i32;
    // SAFETY: `idx` holds as many entries as LightGBM samples (its default
    // `bin_construct_sample_cnt`, or every row when there are fewer).
    l.check(unsafe {
        (l.sample_indices)(
            total as i32,
            cparams.as_ptr(),
            idx.as_mut_ptr().cast(),
            &mut got,
        )
    });
    idx.truncate(got as usize);
    // Pass 1: the sampled rows, column by column as LightGBM takes them,
    // and every label.
    let mut src = Source::new(&a.dir);
    let mut sv: Vec<Vec<f64>> = vec![Vec::new(); p];
    let mut si: Vec<Vec<c_int>> = vec![Vec::new(); p];
    let mut labels: Vec<f32> = Vec::with_capacity(total);
    let mut start = 0usize;
    let mut k = 0usize;
    let value = |v: &View, cuts: &[Vec<f64>], j: usize, i: usize| -> f64 {
        if j == nx {
            return f64::from(v.cat[i]);
        }
        let (vals, nulls) = v.x[j];
        if !nulls.is_none_or(|nb| nb.is_valid(i)) {
            return f64::NAN;
        }
        if cuts.is_empty() || feed != "codes" {
            vals[i]
        } else {
            cuts[j].partition_point(|&e| e <= vals[i]) as f64
        }
    };
    while let Some(b) = src.next() {
        let v = view(&b, nx);
        while k < idx.len() && (idx[k] as usize) < start + v.n {
            let i = idx[k] as usize - start;
            for j in 0..p {
                let x = value(&v, &cuts, j, i);
                if x.abs() > 1e-35 || x.is_nan() {
                    sv[j].push(x);
                    si[j].push(k as c_int);
                }
            }
            k += 1;
        }
        labels.extend(v.y.iter().map(|&y| y as f32));
        start += v.n;
    }
    assert_eq!(start, total, "--rows does not match the files");
    let mut ds: Handle = std::ptr::null_mut();
    let mut vptr: Vec<*mut f64> = sv.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let mut iptr: Vec<*mut c_int> = si.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let per: Vec<c_int> = sv.iter().map(|v| v.len() as c_int).collect();
    ph.next("bin");
    // SAFETY: the column pointers stay alive across the call.
    l.check(unsafe {
        (l.from_sampled)(
            vptr.as_mut_ptr(),
            iptr.as_mut_ptr(),
            p as i32,
            per.as_ptr(),
            idx.len() as i32,
            total as i32,
            total as i64,
            cparams.as_ptr(),
            &mut ds,
        )
    });
    // Pass 2: each batch converted to rows in parallel and pushed.
    src.reset();
    let (mut conv_s, mut push_s, mut row) = (0.0, 0.0, 0i32);
    while let Some(b) = src.next() {
        let t = Instant::now();
        let v = view(&b, nx);
        let n = v.n as i32;
        let (buf32, buf64): (Vec<f32>, Vec<f64>) = match feed.as_str() {
            "f64" => (Vec::new(), rows(&v, |x| x, f64::NAN)),
            "f32" => (rows(&v, |x| x as f32, f32::NAN), Vec::new()),
            "codes" => {
                let cs = codes(&v, &cuts);
                let mut out = vec![f32::NAN; v.n * p];
                out.par_chunks_mut(BLOCK * p)
                    .enumerate()
                    .for_each(|(bi, chunk)| {
                        let r0 = bi * BLOCK;
                        for r in 0..chunk.len() / p {
                            for (j, col) in cs.iter().enumerate() {
                                let code = col[r0 + r];
                                if code != MISSING_CODE {
                                    chunk[r * p + j] = f32::from(code);
                                }
                            }
                            chunk[r * p + p - 1] = f32::from(v.cat[r0 + r]);
                        }
                    });
                (out, Vec::new())
            }
            f => panic!("feed {f}"),
        };
        conv_s += t.elapsed().as_secs_f64();
        let t = Instant::now();
        // SAFETY: the buffer holds `n * p` values of the type named.
        l.check(unsafe {
            if buf64.is_empty() {
                (l.push_rows)(ds, buf32.as_ptr().cast(), 0, n, p as i32, row)
            } else {
                (l.push_rows)(ds, buf64.as_ptr().cast(), 1, n, p as i32, row)
            }
        });
        push_s += t.elapsed().as_secs_f64();
        row += n;
    }
    // SAFETY: `labels` holds `total` float32 values.
    l.check(unsafe {
        (l.set_field)(
            ds,
            c("label").as_ptr(),
            labels.as_ptr().cast(),
            total as c_int,
            0,
        )
    });
    ph.next("boost");
    let mut bst: Handle = std::ptr::null_mut();
    // SAFETY: handles live until the process ends.
    unsafe {
        l.check((l.booster_create)(ds, cparams.as_ptr(), &mut bst));
        let mut empty: c_int = 0;
        for _ in 0..rounds {
            l.check((l.update)(bst, &mut empty));
        }
        l.check((l.save)(
            bst,
            0,
            -1,
            0,
            c(a.get("model").expect("--model")).as_ptr(),
        ));
    }
    format!(
        "{{\"cmd\": \"lgb\", \"read_s\": {:.2}, \"convert_s\": {conv_s:.2}, \"push_s\": {push_s:.2}, \"source_passes\": {}, {}}}",
        src.read_s,
        src.passes,
        ph.json()
    )
}

fn main() {
    let a = Args::parse();
    if let Some(t) = a.get("threads") {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t.parse().unwrap())
            .build_global()
            .unwrap();
    }
    let out = match a.cmd.as_str() {
        "sketch" => sketch(&a),
        "xgb" => xgb(&a),
        "lgb" => lgb(&a),
        other => panic!("unknown command {other}"),
    };
    println!("{out}");
}
