//! XGBoost's and LightGBM's C functions, loaded at run time from the shared
//! library each Python package ships. Nothing here is linked (CLAUDE.md,
//! rule 12): `libloading` opens the file by the path the Python package
//! loaded it from, and the operating system hands back the image already in
//! the process, with its state, so that a handle made here is the Python
//! package's to use and to free.
//!
//! Each function's type is the one its header declares, at XGBoost 3.4.1
//! (`include/xgboost/c_api.h`) and LightGBM 4.7.0
//! (`include/LightGBM/c_api.h`); a library that lacks one is refused by
//! name, as is a version outside the range the native engine is built for.

use std::ffi::{CStr, c_char, c_int, c_void};

use libloading::Library;

use super::GbdtError;

pub(crate) type Handle = *mut c_void;

/// The XGBoost versions the native engine is built for: the C API it calls
/// is the same from 2.0 through 3.x (measured on 3.4.1).
pub const XGB_VERSIONS: ((i32, i32), (i32, i32)) = ((2, 0), (4, 0));
/// The LightGBM versions the native engine is built for (measured on 4.7.0).
pub const LGB_VERSIONS: ((i32, i32), (i32, i32)) = ((4, 0), (5, 0));

fn open(path: &str, what: &str) -> Result<Library, GbdtError> {
    // SAFETY: opening a shared library runs its initializers. The path is
    // the one the library's own Python package has already loaded, so the
    // image is in the process and its initializers have run.
    unsafe { Library::new(path) }
        .map_err(|e| GbdtError::msg(format!("{what} could not be loaded from {path}: {e}")))
}

macro_rules! symbols {
    ($lib:ident, $what:literal, $path:ident; $($field:ident = $name:literal),* $(,)?) => {
        $(
            // SAFETY: the type the struct declares for the field is the one
            // the library's header declares for the symbol.
            let $field = *unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.map_err(|_| {
                GbdtError::msg(format!(
                    "{} at {} has no `{}`: it is not a version whose C API po.gbdt's native \
                     engine calls; engine=\"python\" builds through the library's Python API",
                    $what, $path, $name
                ))
            })?;
        )*
    };
}

/// The XGBoost C functions the native engine calls.
#[allow(clippy::type_complexity)]
pub struct XgbLib {
    pub(crate) last_error: unsafe extern "C" fn() -> *const c_char,
    pub(crate) proxy_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    pub(crate) from_callback: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        extern "C" fn(Handle),
        extern "C" fn(Handle) -> c_int,
        *const c_char,
        *mut Handle,
    ) -> c_int,
    pub(crate) set_columnar: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
    pub(crate) set_info: unsafe extern "C" fn(Handle, *const c_char, *const c_char) -> c_int,
    pub(crate) set_str_info:
        unsafe extern "C" fn(Handle, *const c_char, *const *const c_char, u64) -> c_int,
    pub(crate) from_dense: unsafe extern "C" fn(*const c_char, *const c_char, *mut Handle) -> c_int,
    pub(crate) free: unsafe extern "C" fn(Handle) -> c_int,
    pub(crate) num_row: unsafe extern "C" fn(Handle, *mut u64) -> c_int,
    version: (i32, i32, i32),
    _lib: Library,
}

impl XgbLib {
    /// XGBoost's library at `path` (`xgboost.core._LIB.path`), refused by
    /// name when a function is missing or its version is outside
    /// [`XGB_VERSIONS`].
    pub fn load(path: &str) -> Result<Self, GbdtError> {
        let lib = open(path, "XGBoost's library")?;
        symbols!(lib, "XGBoost's library", path;
            version_fn = "XGBoostVersion",
        );
        let version_fn: unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int) = version_fn;
        let (mut a, mut b, mut c) = (0, 0, 0);
        // SAFETY: three integers for XGBoost to fill.
        unsafe { version_fn(&mut a, &mut b, &mut c) };
        let ((lo_a, lo_b), (hi_a, hi_b)) = XGB_VERSIONS;
        if (a, b) < (lo_a, lo_b) || (a, b) >= (hi_a, hi_b) {
            return Err(GbdtError::msg(format!(
                "XGBoost {a}.{b}.{c} at {path}: po.gbdt's native engine calls the C API of \
                 XGBoost {lo_a}.{lo_b} up to {hi_a}.{hi_b} (not included); engine=\"python\" \
                 builds through the library's Python API"
            )));
        }
        symbols!(lib, "XGBoost's library", path;
            last_error = "XGBGetLastError",
            proxy_create = "XGProxyDMatrixCreate",
            from_callback = "XGQuantileDMatrixCreateFromCallback",
            set_columnar = "XGProxyDMatrixSetDataColumnar",
            set_info = "XGDMatrixSetInfoFromInterface",
            set_str_info = "XGDMatrixSetStrFeatureInfo",
            from_dense = "XGDMatrixCreateFromDense",
            free = "XGDMatrixFree",
            num_row = "XGDMatrixNumRow",
        );
        Ok(Self {
            last_error,
            proxy_create,
            from_callback,
            set_columnar,
            set_info,
            set_str_info,
            from_dense,
            free,
            num_row,
            version: (a, b, c),
            _lib: lib,
        })
    }

    /// The library's version, `(major, minor, patch)`.
    pub fn version(&self) -> (i32, i32, i32) {
        self.version
    }

    /// `rc` as a result, with XGBoost's own message on failure.
    pub(crate) fn check(&self, rc: c_int, doing: &str) -> Result<(), GbdtError> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: XGBoost returns a NUL-terminated message it owns, valid
        // until its next call on this thread.
        let msg = unsafe { CStr::from_ptr((self.last_error)()) };
        Err(GbdtError::msg(format!(
            "XGBoost, {doing}: {}",
            first_line(&msg.to_string_lossy())
        )))
    }
}

/// The LightGBM C functions the native engine calls.
#[allow(clippy::type_complexity)]
pub struct LgbLib {
    pub(crate) last_error: unsafe extern "C" fn() -> *const c_char,
    pub(crate) sample_count: unsafe extern "C" fn(i32, *const c_char, *mut c_int) -> c_int,
    pub(crate) sample_indices:
        unsafe extern "C" fn(i32, *const c_char, *mut c_void, *mut i32) -> c_int,
    pub(crate) from_sampled: unsafe extern "C" fn(
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
    pub(crate) by_reference: unsafe extern "C" fn(Handle, i64, *mut Handle) -> c_int,
    pub(crate) push_rows:
        unsafe extern "C" fn(Handle, *const c_void, c_int, i32, i32, i32) -> c_int,
    pub(crate) set_field:
        unsafe extern "C" fn(Handle, *const c_char, *const c_void, c_int, c_int) -> c_int,
    pub(crate) set_names: unsafe extern "C" fn(Handle, *const *const c_char, c_int) -> c_int,
    pub(crate) free: unsafe extern "C" fn(Handle) -> c_int,
    pub(crate) num_data: unsafe extern "C" fn(Handle, *mut c_int) -> c_int,
    /// `LGBM_DatasetCreateFromArrowStream`, from LightGBM 4.7.0 on.
    pub(crate) from_arrow:
        Option<unsafe extern "C" fn(*mut c_void, *const c_char, Handle, *mut Handle) -> c_int>,
    _lib: Library,
}

impl LgbLib {
    /// LightGBM's library at `path` (`lightgbm.basic._LIB` was loaded from
    /// it), `version` being `lightgbm.__version__`; refused by name when a
    /// function is missing or the version is outside [`LGB_VERSIONS`].
    pub fn load(path: &str, version: &str) -> Result<Self, GbdtError> {
        let parsed = parse_version(version);
        let ((lo_a, lo_b), (hi_a, hi_b)) = LGB_VERSIONS;
        match parsed {
            Some(v) if v >= (lo_a, lo_b) && v < (hi_a, hi_b) => {}
            _ => {
                return Err(GbdtError::msg(format!(
                    "LightGBM {version}: po.gbdt's native engine calls the C API of LightGBM \
                     {lo_a}.{lo_b} up to {hi_a}.{hi_b} (not included); engine=\"python\" builds \
                     through the library's Python API"
                )));
            }
        }
        let lib = open(path, "LightGBM's library")?;
        symbols!(lib, "LightGBM's library", path;
            last_error = "LGBM_GetLastError",
            sample_count = "LGBM_GetSampleCount",
            sample_indices = "LGBM_SampleIndices",
            from_sampled = "LGBM_DatasetCreateFromSampledColumn",
            by_reference = "LGBM_DatasetCreateByReference",
            push_rows = "LGBM_DatasetPushRows",
            set_field = "LGBM_DatasetSetField",
            set_names = "LGBM_DatasetSetFeatureNames",
            free = "LGBM_DatasetFree",
            num_data = "LGBM_DatasetGetNumData",
        );
        // SAFETY: as in `symbols!`; an older LightGBM simply lacks it.
        let from_arrow = unsafe { lib.get(b"LGBM_DatasetCreateFromArrowStream\0") }
            .ok()
            .map(|s| *s);
        Ok(Self {
            last_error,
            sample_count,
            sample_indices,
            from_sampled,
            by_reference,
            push_rows,
            set_field,
            set_names,
            free,
            num_data,
            from_arrow,
            _lib: lib,
        })
    }

    /// Whether LightGBM takes a whole Arrow stream (4.7.0 on).
    pub fn has_arrow(&self) -> bool {
        self.from_arrow.is_some()
    }

    pub(crate) fn check(&self, rc: c_int, doing: &str) -> Result<(), GbdtError> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: LightGBM returns a NUL-terminated message it owns.
        let msg = unsafe { CStr::from_ptr((self.last_error)()) };
        Err(GbdtError::msg(format!(
            "LightGBM, {doing}: {}",
            first_line(&msg.to_string_lossy())
        )))
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s).trim()
}

/// `(major, minor)` of a version string such as `4.7.0` or `4.7.0.dev1`.
fn parse_version(v: &str) -> Option<(i32, i32)> {
    let mut it = v.split('.');
    let a = it.next()?.trim().parse().ok()?;
    let b: String = it
        .next()
        .unwrap_or("0")
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    Some((a, b.parse().unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_parse() {
        assert_eq!(parse_version("4.7.0"), Some((4, 7)));
        assert_eq!(parse_version("4.10rc1"), Some((4, 10)));
        assert_eq!(parse_version("5"), Some((5, 0)));
        assert_eq!(parse_version("x"), None);
    }

    #[test]
    fn a_missing_library_is_refused_by_path() {
        let e = XgbLib::load("/nonexistent/libxgboost.so").err().unwrap();
        assert!(e.to_string().contains("/nonexistent/libxgboost.so"), "{e}");
        let e = LgbLib::load("/nonexistent/lib_lightgbm.so", "4.7.0")
            .err()
            .unwrap();
        assert!(
            e.to_string().contains("/nonexistent/lib_lightgbm.so"),
            "{e}"
        );
    }

    #[test]
    fn a_lightgbm_version_outside_the_range_is_refused_before_loading() {
        for v in ["3.3.5", "5.0.0", "garbage"] {
            let e = LgbLib::load("/nonexistent/lib_lightgbm.so", v)
                .err()
                .unwrap();
            assert!(e.to_string().contains(&format!("LightGBM {v}")), "{e}");
        }
    }
}
