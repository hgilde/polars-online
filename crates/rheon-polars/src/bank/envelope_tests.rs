//! The file's format version: 3 only when a spec carries a duration, so
//! every other file stays readable by a build from before durations. A file
//! of its own, keeping bank.rs under `tests/test_repo_hygiene.py`'s 250 KB
//! cap for a source file.
use super::{BankHeader, format_version_for};
use crate::{Bank, Spec};

fn spec(extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x"]{extra}}}"#
    ))
    .unwrap()
}

/// A file is version 3 only when a spec carries a duration (task 88), so
/// every other file stays readable by a build from before durations.
#[test]
fn a_file_says_version_3_only_when_a_spec_carries_a_duration() {
    for (extra, version) in [
        (r#", "half_life": 50"#, 2),
        (r#", "clock": "t", "half_life": 600, "gap_cap": 300"#, 2),
        (r#", "clock": "t", "half_life": "inf", "gap_cap": 1e12"#, 2),
        (r#", "clock": "t", "half_life": "10m", "gap_cap": "5m""#, 3),
    ] {
        let specs = vec![spec(extra)];
        assert_eq!(format_version_for(&specs), version, "{extra}");
        let bytes = Bank::new(specs).unwrap().save_bytes().unwrap();
        let header: BankHeader = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(header.format_version, version, "{extra}");
        // And the file loads back, specs and all.
        let back = Bank::load_bytes(&bytes, None).unwrap();
        assert_eq!(back.specs()[0].half_life, spec(extra).half_life, "{extra}");
    }
}
