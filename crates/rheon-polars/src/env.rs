//! The environment variables the library reads, each by its `RHEON_` name and,
//! until the next major release, by the name it had before the library was
//! renamed.
//!
//! A variable is read by its new name. Only when that is not set at all (an
//! empty value is set) is the old name read, and then the caller is told
//! which name the value came from, so that an error can quote the variable
//! the user typed. When both are set the new one wins and the old one is
//! never looked at again.
//!
//! The library prints nothing here: Python warns once for each old name at
//! import, and the command line prints one notice for each at its start,
//! from [`legacy_notices`].

use std::ffi::OsString;

/// One variable, by its current name and the name it had before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var {
    /// The current name, `RHEON_...`.
    pub name: &'static str,
    /// The name before the rename, read when `name` is not set.
    pub old: &'static str,
}

/// The size of the bank's thread pool.
pub const MAX_THREADS: Var = Var {
    name: "RHEON_MAX_THREADS",
    old: "POLARS_ONLINE_MAX_THREADS",
};

/// `1` prints where a run's time goes to standard error.
pub const TIMING: Var = Var {
    name: "RHEON_TIMING",
    old: "POLARS_ONLINE_TIMING",
};

/// `1` turns the unstable label's warning on.
pub const WARN_UNSTABLE: Var = Var {
    name: "RHEON_WARN_UNSTABLE",
    old: "POLARS_ONLINE_WARN_UNSTABLE",
};

/// Every variable the Rust side reads. (Python also reads
/// `RHEON_SKIP_CPU_CHECK`, which Rust never does.)
pub const READ_BY_RUST: [Var; 3] = [MAX_THREADS, TIMING, WARN_UNSTABLE];

impl Var {
    /// The value and the name it was read from: the current name if it is
    /// set, else the old one, else `None`.
    #[must_use]
    pub fn read(&self) -> Option<(&'static str, OsString)> {
        if let Some(v) = std::env::var_os(self.name) {
            return Some((self.name, v));
        }
        std::env::var_os(self.old).map(|v| (self.old, v))
    }

    /// The value, whichever name it came from.
    #[must_use]
    pub fn value(&self) -> Option<OsString> {
        self.read().map(|(_, v)| v)
    }

    /// What a process is told when it set the old name and not the new one,
    /// or `None` when it did not.
    #[must_use]
    pub fn legacy_notice(&self) -> Option<String> {
        if std::env::var_os(self.name).is_none() && std::env::var_os(self.old).is_some() {
            Some(format!(
                "{} was renamed {}, and the old name is read for now. Set {} instead.",
                self.old, self.name, self.name
            ))
        } else {
            None
        }
    }
}

/// The notice for each variable of [`READ_BY_RUST`] that is set by its old
/// name alone: what the command line prints once, at its start.
#[must_use]
pub fn legacy_notices() -> Vec<String> {
    READ_BY_RUST.iter().filter_map(Var::legacy_notice).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests read and write the process's environment, so each takes
    // names of its own: no other test in the binary touches them.
    const A: Var = Var {
        name: "RHEON_TEST_ENV_A",
        old: "POLARS_ONLINE_TEST_ENV_A",
    };

    #[test]
    fn the_new_name_wins_and_the_old_is_read_only_when_it_is_unset() {
        // SAFETY: names of this test alone, and no other thread reads them.
        unsafe {
            std::env::remove_var(A.name);
            std::env::remove_var(A.old);
        }
        assert_eq!(A.read(), None);
        assert_eq!(A.legacy_notice(), None);

        unsafe { std::env::set_var(A.old, "old") };
        assert_eq!(A.read(), Some((A.old, OsString::from("old"))));
        let notice = A.legacy_notice().unwrap();
        assert!(
            notice.contains(A.old) && notice.contains(A.name),
            "{notice}"
        );

        unsafe { std::env::set_var(A.name, "new") };
        assert_eq!(A.read(), Some((A.name, OsString::from("new"))));
        assert_eq!(A.value(), Some(OsString::from("new")));
        assert_eq!(A.legacy_notice(), None, "both set: no notice");

        // An empty new value is set, and wins over a set old one.
        unsafe { std::env::set_var(A.name, "") };
        assert_eq!(A.value(), Some(OsString::new()));

        unsafe {
            std::env::remove_var(A.name);
            std::env::remove_var(A.old);
        }
    }

    #[test]
    fn the_three_variables_the_rust_side_reads_are_named_as_documented() {
        let names: Vec<_> = READ_BY_RUST.iter().map(|v| (v.name, v.old)).collect();
        assert_eq!(
            names,
            [
                ("RHEON_MAX_THREADS", "POLARS_ONLINE_MAX_THREADS"),
                ("RHEON_TIMING", "POLARS_ONLINE_TIMING"),
                ("RHEON_WARN_UNSTABLE", "POLARS_ONLINE_WARN_UNSTABLE"),
            ]
        );
    }
}
