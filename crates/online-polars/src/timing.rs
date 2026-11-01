//! Where a run's time goes, printed to standard error under
//! `POLARS_ONLINE_TIMING=1`.
//!
//! The bank prints one line per chunk, its sections in milliseconds, and the
//! runner one line per run, its waits on the reader and the writer in
//! seconds. Those lines are for a person reading them: their words and
//! their order are not promised, and a release may change them.
//!
//! The variable is on at `1` and nothing else, as `POLARS_ONLINE_WARN_UNSTABLE`
//! is, and as Polars reads its own `POLARS_WARN_UNSTABLE`. Its old name,
//! `ONLINE_TIMING`, turns nothing on: set, it gets one notice per process,
//! naming the new one (task 244, C-1).
//!
//! Both are read once, when the process first asks, and the answer kept: a
//! variable set or cleared after that changes nothing until the next
//! process. Reading the environment costs in proportion to its size, and a
//! read on every chunk moved the instruction count of a bank's call with
//! whatever else the process's environment held (task 292, P8).

use std::sync::OnceLock;

/// The environment variable that turns the timings on.
pub const TIMING_VAR: &str = "POLARS_ONLINE_TIMING";

/// The variable's old name, read only to say it was renamed.
const OLD_TIMING_VAR: &str = "ONLINE_TIMING";

/// What a process with the old name set is told, once.
pub fn renamed_timing_notice() -> String {
    format!("{OLD_TIMING_VAR} was renamed {TIMING_VAR}; set {TIMING_VAR}=1 for the timings")
}

/// Whether to print the timings: [`TIMING_VAR`] is `1` in the process's
/// environment as the first call found it.
pub fn timing_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os(OLD_TIMING_VAR).is_some() {
            eprintln!("polars-online: warning: {}", renamed_timing_notice());
        }
        std::env::var(TIMING_VAR).is_ok_and(|v| v == "1")
    })
}
