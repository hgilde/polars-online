//! Records the x86 CPU features this build was compiled to need, in
//! `python/polars_online/_build_flags.py`, for the import-time CPU check
//! (`python/polars_online/_cpu_check.py`).
//!
//! The wheels of `polars-online` for x86-64 are built with
//! `-C target-feature=+avx2,+fma,...` (release.yml), so the compiler may
//! emit instructions an older CPU does not have, and that CPU would die on
//! an illegal instruction. The check runs in Python before the extension
//! module is loaded, and it needs to know what to look for. Reading the
//! features here, from the flags cargo hands rustc, makes the record the
//! flags themselves: a build cannot name one set of features and compile
//! for another. Polars records its own runtime wheels' features the same way
//! (`py-polars/runtime/polars-runtime-32/build.rs`).
//!
//! A build with no `target-feature` flag, and any build that is not for x86,
//! records the empty string, which is no check. The file is not committed:
//! it is written here, and a missing one is written again (the
//! `rerun-if-changed` on it below).
//!
//! Standard library only, so a test compiles this file with `rustc` and runs
//! it (tests/test_cpu_check.py).

use std::env;
use std::fs;
use std::path::PathBuf;

/// The features `-C target-feature=...` turns on in a `CARGO_ENCODED_RUSTFLAGS`
/// value (the flags joined by `0x1f`), in order. Features it turns off, and
/// `crt-static`, which is a linking choice and no CPU feature, are left out.
fn required_features(encoded: &str) -> Vec<String> {
    let flags: Vec<&str> = encoded.split('\u{1f}').filter(|f| !f.is_empty()).collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        let flag = flags[i];
        // `-C target-feature=...` (two words), `-Ctarget-feature=...`,
        // `--codegen target-feature=...` and `--codegen=target-feature=...`.
        let value = if flag == "-C" || flag == "--codegen" {
            i += 1;
            flags.get(i).copied()
        } else if let Some(rest) = flag.strip_prefix("--codegen=") {
            Some(rest)
        } else {
            flag.strip_prefix("-C")
        };
        if let Some(list) = value.and_then(|v| v.strip_prefix("target-feature=")) {
            out.extend(
                list.split(',')
                    .filter_map(|feature| feature.strip_prefix('+'))
                    .filter(|name| !name.is_empty() && *name != "crt-static")
                    .map(str::to_string),
            );
        }
        i += 1;
    }
    out
}

/// The text of `_build_flags.py` for these features.
fn render(features: &[String]) -> String {
    let list: Vec<String> = features.iter().map(|f| format!("+{f}")).collect();
    format!(
        "# Written by crates/online-py/build.rs; not committed.\n\
         # The x86 CPU features this build was compiled to need, as rustc was told,\n\
         # empty for a build with none (the CPU check then does nothing).\n\
         BUILD_FEATURE_FLAGS = \"{}\"\n",
        list.join(",")
    )
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/online-py/ -> the workspace root, where the Python source sits
    // (also in an sdist, which keeps the layout).
    let target = manifest_dir
        .join("..")
        .join("..")
        .join("python")
        .join("polars_online")
        .join("_build_flags.py");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    // A file that is missing counts as changed, so a checkout that never had
    // one, or whose file was deleted, writes it even from a cached build.
    println!("cargo:rerun-if-changed={}", target.display());

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = if arch == "x86_64" || arch == "x86" {
        required_features(&env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default())
    } else {
        Vec::new()
    };
    let text = render(&features);

    // The Python source folder is the one place maturin reads from. A layout
    // without it (the crate built on its own) records nothing, which is no
    // check, and is no reason to fail the build.
    let Some(dir) = target.parent().filter(|d| d.is_dir()) else {
        return;
    };
    // Only when it differs, so a build that records the same flags leaves the
    // file's time alone and cargo does not run this again for it.
    if !fs::read_to_string(&target).is_ok_and(|old| old == text) {
        fs::write(dir.join("_build_flags.py"), text)
            .unwrap_or_else(|e| panic!("could not write {}: {e}", target.display()));
    }
}
