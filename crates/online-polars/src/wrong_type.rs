//! The refusal of a value or a column of the wrong type, told apart from
//! every other refusal by its kind, not its text (task 355).
//!
//! A caller that raises a class of its own for a wrong type -- the Python
//! extension raises `polars_online.WrongTypeError`, a `TypeError` -- reads
//! the kind here. The text is the same either way, so the CLI, which prints
//! the text, says what it said before.
//!
//! The kind is Polars' own `InvalidOperation`, the error Polars gives an
//! operation a dtype cannot take. Its display is the message alone, as
//! `ComputeError`'s is, so no message changes. An `InvalidOperation` that
//! comes from Polars itself through `?` is a dtype refusal too, and reads
//! as one.

use std::fmt;
use std::ops::Deref;

use polars::prelude::PolarsError;

/// The error refusing a value or a column of the wrong type, `msg` its
/// whole text.
pub fn wrong_type(msg: impl Into<String>) -> PolarsError {
    PolarsError::InvalidOperation(msg.into().into())
}

/// Whether `e` refuses a value or a column of the wrong type
/// ([`wrong_type`]), through any context wrapped around it.
pub fn is_wrong_type(e: &PolarsError) -> bool {
    match e {
        PolarsError::InvalidOperation(_) => true,
        PolarsError::Context { error, .. } | PolarsError::ExprContext { error, .. } => {
            is_wrong_type(error)
        }
        _ => false,
    }
}

/// A refusal as text, with its kind: what a call checked before any row is
/// read returns ([`crate::WindowsRun::new`]). It reads as the `str` it
/// holds, so a caller that wants only the text has it.
#[derive(Clone, PartialEq, Eq)]
pub struct Refused {
    msg: String,
    wrong_type: bool,
}

impl Refused {
    /// The refusal of a value or a column of the wrong type.
    pub fn wrong_type(msg: impl Into<String>) -> Self {
        Self {
            msg: msg.into(),
            wrong_type: true,
        }
    }

    /// Whether this refuses a value or a column of the wrong type.
    pub fn is_wrong_type(&self) -> bool {
        self.wrong_type
    }

    /// The text.
    pub fn into_string(self) -> String {
        self.msg
    }
}

/// Any other refusal.
impl From<String> for Refused {
    fn from(msg: String) -> Self {
        Self {
            msg,
            wrong_type: false,
        }
    }
}

impl From<Refused> for String {
    fn from(r: Refused) -> Self {
        r.msg
    }
}

impl Deref for Refused {
    type Target = str;
    fn deref(&self) -> &str {
        &self.msg
    }
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

/// As the text's: an `unwrap` prints what a `String` error printed.
impl fmt::Debug for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.msg, f)
    }
}

impl std::error::Error for Refused {}

impl PartialEq<str> for Refused {
    fn eq(&self, other: &str) -> bool {
        self.msg == other
    }
}

impl PartialEq<&str> for Refused {
    fn eq(&self, other: &&str) -> bool {
        self.msg == *other
    }
}

impl PartialEq<String> for Refused {
    fn eq(&self, other: &String) -> bool {
        self.msg == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kind_survives_a_wrapped_message_and_a_context() {
        let e = wrong_type("spec \"m\": feature column \"x\" must be a number");
        assert!(is_wrong_type(&e));
        assert_eq!(
            e.to_string(),
            "spec \"m\": feature column \"x\" must be a number"
        );
        assert!(is_wrong_type(
            &e.wrap_msg(|m| format!("input a.parquet: {m}"))
        ));
        assert!(is_wrong_type(&e.context("chunk 2".into())));
        assert!(!is_wrong_type(&PolarsError::ComputeError("x".into())));
        assert!(!is_wrong_type(&PolarsError::SchemaMismatch("x".into())));
    }

    #[test]
    fn a_refusal_reads_as_its_text() {
        let r = Refused::wrong_type("with_windows: f: ewm_mean's input must be a number");
        assert!(r.is_wrong_type());
        assert!(r.starts_with("with_windows:"));
        assert_eq!(format!("{r}"), format!("{r:?}").trim_matches('"'));
        let other = Refused::from("with_windows: no expression".to_string());
        assert!(!other.is_wrong_type());
        assert_eq!(String::from(other), "with_windows: no expression");
    }
}
