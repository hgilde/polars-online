//! How a bank file is encoded and checked: the groups' states written lazily,
//! and the JSON export read back.

use super::*;

/// The lines around a parse failure, so the error names the field rather than
/// a line number nobody can look up: the export is thousands of lines and the
/// enclosing keys are what identify the struct.
pub(super) fn near(text: &str, line: usize) -> String {
    if line == 0 {
        return String::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    let lo = line.saturating_sub(4);
    let hi = line.min(lines.len());
    let mut out = String::from(", at:");
    for (i, l) in lines[lo..hi].iter().enumerate() {
        let n = lo + i + 1;
        let mark = if n == line { ">" } else { " " };
        out.push_str(&format!("\n  {mark} {}", l.trim_end()));
    }
    out
}

/// Is `text` the whole of `file`? The JSON is read back into a `BankFile`
/// and re-encoded, and the msgpack must match the real state byte for byte.
///
/// This is the enforcement behind [`online_core::humanfloat`], not a
/// belt-and-braces check. JSON has no literal for `NaN` or `±inf` and
/// `serde_json` writes all three as `null` without a word, so every config
/// float a caller may set to infinity needs annotating -- and the only way
/// to know one was missed is to read the export back. A `null` where an
/// `f64` belongs fails to parse, and the byte comparison catches anything
/// subtler. `O(state)` twice over, paid once at the end of a run.
///
/// Note that the interception is in `serde_json`'s *serializer*, not its
/// formatter: `serialize_f64` sends a non-finite value to `write_null`
/// before any `Formatter` sees it, so a custom formatter cannot fix this
/// and the annotation has to be on the field.
pub(super) fn json_is_faithful(file: &BankFile, text: &str) -> Result<(), String> {
    let want = rmp_serde::to_vec_named(file).map_err(|e| e.to_string())?;
    let back: BankFile = serde_json::from_str(text).map_err(|e| {
        format!(
            // A field that writes NaN or ±inf as null, which needs
            // `#[serde(with = "crate::humanfloat::f64_or_tag")]` in online-core.
            "ModelBank.write_json: the export dropped a value the state holds ({e}), which is a \
             defect in polars-online. Open an issue at \
             https://github.com/hgilde/polars-online/issues with this message and the lines it \
             shows{}",
            near(text, e.line())
        )
    })?;
    let got = rmp_serde::to_vec_named(&back).map_err(|e| e.to_string())?;
    if got != want {
        return Err(
            "ModelBank.write_json: the export did not read back as the state it came from, \
             which is a defect in polars-online. Open an issue at \
             https://github.com/hgilde/polars-online/issues with this message."
                .to_string(),
        );
    }
    Ok(())
}

/// The groups' states, snapshotted one group at a time as they are encoded,
/// rather than all at once beforehand: the file's bytes are the same, and a
/// save holds one group's snapshot beside the bank instead of a copy of it.
pub(super) struct LazyStates<'a>(pub(super) &'a Bank);

impl Serialize for LazyStates<'_> {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        struct Spec<'a>(Vec<(&'a GroupKey, &'a Stream)>);
        impl Serialize for Spec<'_> {
            fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeSeq;
                let mut seq = ser.serialize_seq(Some(self.0.len()))?;
                for (k, st) in &self.0 {
                    // A tuple of a reference and the snapshot writes the
                    // same two-element array `(GroupKey, StreamState)` does.
                    seq.serialize_element(&(*k, st.save()))?;
                }
                seq.end()
            }
        }
        let mut seq = ser.serialize_seq(Some(self.0.states.len()))?;
        for hm in &self.0.states {
            let mut v: Vec<(&GroupKey, &Stream)> = hm.iter().collect();
            v.sort_by(|a, b| a.0.cmp(b.0));
            seq.serialize_element(&Spec(v))?;
        }
        seq.end()
    }
}
