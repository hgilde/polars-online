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
/// This is the enforcement behind [`rheon_core::humanfloat`], not a
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
            // `#[serde(with = "crate::humanfloat::f64_or_tag")]` in rheon-core.
            "ModelBank.write_json: the export dropped a value the state holds ({e}), which is a \
             defect in rheon. Open an issue at \
             https://github.com/rheon-dev/rheon/issues with this message and the lines it \
             shows{}",
            near(text, e.line())
        )
    })?;
    let got = rmp_serde::to_vec_named(&back).map_err(|e| e.to_string())?;
    if got != want {
        return Err(
            "ModelBank.write_json: the export did not read back as the state it came from, \
             which is a defect in rheon. Open an issue at \
             https://github.com/rheon-dev/rheon/issues with this message."
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

/// A writer that keeps the last I/O error its target raised, with its kind
/// and OS code, since the encoder reports a failed write as text. The encoder
/// stops at the first error it is handed, so the last one kept is the one
/// that ended the save; an `Interrupted` is not kept, since `write_all`
/// retries it and the save goes on.
pub(super) struct Stash<'a, W: std::io::Write> {
    inner: &'a mut W,
    error: Option<std::io::Error>,
}

impl<'a, W: std::io::Write> Stash<'a, W> {
    pub(super) fn new(inner: &'a mut W) -> Self {
        Self { inner, error: None }
    }

    pub(super) fn take_error(&mut self) -> Option<std::io::Error> {
        self.error.take()
    }

    fn keep(&mut self, e: &std::io::Error) {
        if e.kind() == std::io::ErrorKind::Interrupted {
            return;
        }
        self.error = Some(match e.raw_os_error() {
            Some(code) => std::io::Error::from_raw_os_error(code),
            None => std::io::Error::new(e.kind(), e.to_string()),
        });
    }
}

impl<W: std::io::Write> std::io::Write for Stash<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf).inspect_err(|e| self.keep(e))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush().inspect_err(|e| self.keep(e))
    }
}

#[cfg(test)]
mod tests {
    use super::super::Bank;
    use std::io::{self, ErrorKind, Write};

    struct Failing(ErrorKind, Option<i32>);
    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(match self.1 {
                Some(c) => io::Error::from_raw_os_error(c),
                None => io::Error::new(self.0, "no"),
            })
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A write that fails mid-encode reaches the caller as the same kind
    /// (and OS code), not as the encoder's text.
    #[test]
    fn a_failed_write_keeps_its_error_kind() {
        let spec = serde_json::from_str(
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x"], "half_life": 10.0}"#,
        )
        .unwrap();
        let bank = Bank::new(vec![spec]).unwrap();
        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::StorageFull,
            ErrorKind::NotFound,
        ] {
            let e = bank.write_msgpack_io(&mut Failing(kind, None)).unwrap_err();
            assert_eq!(e.kind(), kind);
        }
        let e = bank
            .write_msgpack_io(&mut Failing(ErrorKind::Other, Some(libc_enospc())))
            .unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc_enospc()));
    }

    /// A write that fails and is retried (`write_all` retries `Interrupted`)
    /// is no error of the save: the one that ends it is. The first error
    /// kept was the retried one, so a save that then met a full disk said
    /// it was interrupted.
    #[test]
    fn a_retried_interruption_is_not_the_error_a_later_failure_is_reported_as() {
        /// Writes `ok` buffers, interrupted once before the first, then fails.
        struct Interrupted {
            fail_with: ErrorKind,
            interrupts: u32,
            writes: u32,
        }
        impl Write for Interrupted {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                if self.interrupts > 0 {
                    self.interrupts -= 1;
                    return Err(io::Error::new(ErrorKind::Interrupted, "signal"));
                }
                self.writes += 1;
                if self.writes > 3 {
                    return Err(io::Error::new(self.fail_with, "the disk"));
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let spec = serde_json::from_str(
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x"], "half_life": 10.0}"#,
        )
        .unwrap();
        let bank = Bank::new(vec![spec]).unwrap();
        let mut w = Interrupted {
            fail_with: ErrorKind::StorageFull,
            interrupts: 1,
            writes: 0,
        };
        let e = bank.write_msgpack_io(&mut w).unwrap_err();
        assert!(w.writes > 3, "the write that fails was reached");
        assert_eq!(e.kind(), ErrorKind::StorageFull, "{e}");
        // Interrupted twice in a row, and every other write after: the same.
        let mut w = Interrupted {
            fail_with: ErrorKind::PermissionDenied,
            interrupts: 2,
            writes: 0,
        };
        let e = bank.write_msgpack_io(&mut w).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::PermissionDenied, "{e}");
    }

    fn libc_enospc() -> i32 {
        if cfg!(windows) { 112 } else { 28 }
    }
}
