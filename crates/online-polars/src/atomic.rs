//! Replace a file, or leave it exactly as it was.
//!
//! `fs::write` truncates the destination and then writes into it, so an
//! interrupted write -- a kill, a full disk, a quota -- leaves a truncated
//! file *and* destroys what was there. For a state file in a `--resume` loop
//! that is the difference between "this save failed, keep going" and "start
//! the stream over": measured, with the write cut a third of the way through,
//! the file is unloadable ("failed to fill whole buffer") and the previous
//! state is gone.
//!
//! So write a temporary sibling, then rename it over the destination. Rust's
//! `fs::rename` "renames a file or directory to a new name, replacing the
//! original file if `to` already exists" on both target platforms (`rename`
//! on Unix, `MoveFileExW` on Windows), and a rename either happens or does
//! not, so a reader sees the old file or the new one and never a half of
//! either.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Numbers the temporaries this process creates, so that two writers of one
/// destination in one process never share one. The pid alone was the name
/// once, and two threads saving the same state file at the same moment --
/// which a plan used twice in one query does, `lf.online.fit_predict(..,
/// save_state=)` under a self-join or `collect_all` -- both created and
/// wrote the same temporary, and the rename published whichever mixture
/// resulted.
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// The temporary for `dest`: a sibling, so the rename stays on one
/// filesystem (`fs::rename` does not cross mount points), named by pid and
/// sequence number, so no two writers -- processes or threads -- share one.
fn temp_name(dest: &Path, seq: u64) -> io::Result<PathBuf> {
    let name = dest.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{}: not a file path", dest.display()),
        )
    })?;
    Ok(dest.with_file_name(format!(
        ".{}.tmp{}-{seq}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

/// A file being written next to `dest`, renamed over it by [`Self::commit`].
///
/// Dropped without committing -- an error on the way, an early `?` -- the
/// temporary is removed and the destination is untouched.
pub(crate) struct AtomicFile {
    dest: PathBuf,
    tmp: PathBuf,
    /// Cleared by `commit`, so `Drop` knows whether to clean up.
    pending: bool,
}

impl AtomicFile {
    /// Create the temporary and hand back the handle to write into. The
    /// caller owns the handle (a `ParquetWriter` consumes one), so `commit`
    /// takes the paths rather than the file.
    pub(crate) fn create(dest: &Path) -> io::Result<(File, Self)> {
        let dest = resolve_symlink(dest);
        let tmp = temp_name(&dest, NEXT_TEMP.fetch_add(1, Ordering::Relaxed))?;
        let file = File::create(&tmp)?;
        Ok((
            file,
            Self {
                dest,
                tmp,
                pending: true,
            },
        ))
    }

    /// Flush the temporary to disk and rename it over the destination.
    ///
    /// The caller's handle is gone by now -- a `ParquetWriter` consumes the
    /// one it is given -- so the sync reopens the file rather than asking for
    /// it back. Reopening is microseconds; the sync is what costs, and
    /// without it the rename can reach the disk while the contents have not,
    /// so a power loss replaces a good state file with an empty one.
    ///
    /// It is not free. On Apple targets std's `sync_all` is
    /// `fcntl(F_FULLFSYNC)` -- and so is `sync_data`, so there is no cheaper
    /// honest option -- which flushes the drive's own cache: measured here,
    /// writing 396 KiB costs 0.10 ms, plain `fsync` 0.13 ms, `F_FULLFSYNC`
    /// 4.0 ms. A save is therefore ~4 ms rather than ~0.5 ms, and a caller
    /// saving after every small chunk should save less often instead. The
    /// alternative is a state file that is not there after the crash it
    /// exists for.
    pub(crate) fn commit(mut self) -> io::Result<()> {
        File::options().write(true).open(&self.tmp)?.sync_all()?;
        fs::rename(&self.tmp, &self.dest)?;
        self.pending = false;
        Ok(())
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if self.pending {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// Write `bytes` to `path`, atomically.
pub(crate) fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let (mut file, pending) = AtomicFile::create(path)?;
    file.write_all(bytes)?;
    drop(file); // Closed, so `commit` reopens exactly one handle to sync.
    pending.commit()
}

/// `fs::write` follows a symlink and writes its target; a rename would
/// replace the link itself. Resolve first, so switching to a rename changes
/// when the destination changes and nothing else. A dangling link has no
/// target to resolve to, and is replaced.
fn resolve_symlink(path: &Path) -> PathBuf {
    match fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => {
            fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_write_leaves_the_destination_alone() {
        let dir = std::env::temp_dir().join(format!("po-atomic-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        fs::write(&dest, b"the good state").unwrap();

        // Make `File::create` on the temporary fail, by putting a directory
        // where it goes. That is the same shape as a full disk: the failure
        // happens on the temporary, before the destination is touched. The
        // name carries a process-wide sequence number, so block the next
        // few hundred: other tests in this binary take some in between.
        let first = NEXT_TEMP.load(Ordering::Relaxed);
        let blockers: Vec<PathBuf> = (first..first + 256)
            .map(|n| temp_name(&dest, n).unwrap())
            .collect();
        for b in &blockers {
            fs::create_dir(b).unwrap();
        }

        let err = write(&dest, b"the new state").unwrap_err();
        assert_eq!(fs::read(&dest).unwrap(), b"the good state", "{err}");

        for b in &blockers {
            fs::remove_dir(b).unwrap();
        }
        write(&dest, b"the new state").unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"the new state");
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "the temporary was left behind"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Two threads saving to one destination at the same moment: each
    /// write succeeds, the file is always exactly one of the two contents,
    /// and no temporary is left. With the pid-only name they shared a
    /// temporary, and the file could be a mixture.
    #[test]
    fn two_writers_of_one_destination_do_not_share_a_temporary() {
        let dir = std::env::temp_dir().join(format!("po-atomic-two-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        let a = vec![b'a'; 200_000];
        let b = vec![b'b'; 200_000];

        let (a, b, dest) = (&a, &b, &dest);
        std::thread::scope(|s| {
            for bytes in [a, b] {
                s.spawn(move || {
                    for _ in 0..50 {
                        write(dest, bytes).unwrap();
                        let got = fs::read(dest).unwrap();
                        assert!(&got == a || &got == b, "a mixture of the two writes");
                    }
                });
            }
        });

        let got = fs::read(dest).unwrap();
        assert!(&got == a || &got == b);
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "a temporary was left behind"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_symlink_is_written_through_not_replaced() {
        let dir = std::env::temp_dir().join(format!("po-atomic-link-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("real.msgpack");
        let link = dir.join("current.msgpack");
        fs::write(&target, b"old").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(not(unix))]
        // Windows needs a privilege for symlinks; without it there is nothing
        // to test, and the resolve path is the same code either way.
        if std::os::windows::fs::symlink_file(&target, &link).is_err() {
            fs::remove_dir_all(&dir).unwrap();
            return;
        }

        write(&link, b"new").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new", "wrote past the link");
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link was replaced by a regular file"
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
