//! The one way a thread's scratch buffer is borrowed.
//!
//! A buffer a function keeps for its thread, in a `thread_local!`
//! `RefCell`, must not stay borrowed while the function runs rayon work: a
//! thread waiting in a rayon scope takes whatever job is queued, which can
//! be another call of the same function on the same thread, and its borrow
//! panics ("RefCell already borrowed"). Two forms are safe. A scratch kept in
//! a `Cell<Option<Box<_>>>` is lent: taken out for the call and set back
//! after it, so nothing is borrowed at all (what [`crate::sandwich`],
//! [`crate::specification`], [`crate::breaks`] and `ewridge` do, for the
//! solves that spread across threads). A scratch kept in a `RefCell` is
//! borrowed through [`with_borrow_mut`] here, for work that runs no parallel
//! section.
//!
//! The second is checked while the tests run: [`with_borrow_mut`] counts the
//! borrows this thread holds, and every parallel section
//! (`pieces::spread`) asserts there are none, at any size or thread count --
//! so a test that reaches a section under a borrow fails on one thread and a
//! small input, without waiting for a steal. And the source is scanned
//! (`no_thread_local_is_borrowed_but_through_this_module`): a raw
//! `with_borrow_mut` or `borrow_mut` anywhere in the library's crates
//! outside this module fails the test, so a new scratch cannot go around the
//! count.

use std::cell::RefCell;
use std::thread::LocalKey;

/// `f` on the thread's buffer in `key`, borrowed for the call. Nothing in
/// `f` may open a parallel section: a test panics if it does.
pub(crate) fn with_borrow_mut<S: 'static, T>(
    key: &'static LocalKey<RefCell<S>>,
    f: impl FnOnce(&mut S) -> T,
) -> T {
    #[cfg(test)]
    let _held = held::Held::enter();
    key.with_borrow_mut(f)
}

/// Called at the head of every parallel section: panics, in the tests, if
/// this thread holds a buffer borrowed through [`with_borrow_mut`].
#[inline]
pub(crate) fn assert_none_borrowed() {
    #[cfg(test)]
    held::assert_none();
}

#[cfg(test)]
mod held {
    use std::cell::Cell;

    thread_local! {
        static HELD: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) struct Held;

    impl Held {
        pub(super) fn enter() -> Self {
            HELD.with(|h| h.set(h.get() + 1));
            Held
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            HELD.with(|h| h.set(h.get() - 1));
        }
    }

    pub(super) fn assert_none() {
        assert_eq!(
            HELD.with(Cell::get),
            0,
            "a parallel section opened while a thread-local buffer is borrowed: a thread \
             waiting in it takes other work, and a call of the same function that runs \
             meanwhile on this thread finds the buffer borrowed. Lend the buffer out \
             (a `Cell<Option<Box<_>>>`: take before, set after) instead"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    thread_local! {
        static BUF: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    }

    /// A parallel section under a borrow is refused, on this one thread and
    /// with a single item: no steal is waited for.
    #[test]
    #[should_panic(expected = "a parallel section opened while a thread-local buffer is borrowed")]
    fn a_parallel_section_under_a_borrow_is_refused() {
        with_borrow_mut(&BUF, |b| {
            b.push(1.0);
            crate::pieces::spread(vec![1], 1, |_| {});
        });
    }

    /// Once the borrow is released, a section runs, and a borrow inside one
    /// item (a piece of a merge borrows its tile) is not a section under it.
    #[test]
    fn a_section_beside_a_borrow_is_not_refused() {
        with_borrow_mut(&BUF, |b| b.push(1.0));
        crate::pieces::spread(vec![1, 2, 3], 1, |_| {
            with_borrow_mut(&BUF, |b| b.push(2.0));
        });
        with_borrow_mut(&BUF, |b| b.clear());
    }

    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("a source directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// No source of the library's crates borrows a thread-local's `RefCell`
    /// mutably but through [`with_borrow_mut`] (`try_borrow_mut`, which
    /// answers a nested call instead of panicking on it, is not one).
    #[test]
    fn no_thread_local_is_borrowed_but_through_this_module() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut files = Vec::new();
        for krate in ["rheon-core", "rheon-polars", "rheon-py", "rheon-cli"] {
            let dir = root.join(krate).join("src");
            if dir.is_dir() {
                sources(&dir, &mut files);
            }
        }
        assert!(files.len() > 100, "{} sources read", files.len());
        let here = Path::new(file!()).file_name().expect("a file name");
        let mut found = Vec::new();
        for path in files {
            if path.file_name() == Some(here) && path.to_string_lossy().contains("rheon-core") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a source");
            for (n, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if code.contains(".with_borrow_mut(") || code.contains(".borrow_mut()") {
                    found.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
        assert!(
            found.is_empty(),
            "borrow a thread-local scratch through `crate::scratch::with_borrow_mut`, or \
             lend it out, as that module's docs say:\n{}",
            found.join("\n")
        );
    }
}
