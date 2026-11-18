//! A fixed-seed hasher for the maps a run keys by group.
//!
//! `std`'s default `RandomState` draws its keys from the operating system once
//! per process, so the probe sequences of a table of group keys differ from one
//! process to the next, and so does the work a run does on it: two runs of one
//! binary on one input count different instructions. [`FixedState`] hashes the
//! same key to the same value in every process on every platform, so a run's
//! cost is a function of its input alone.
//!
//! The trade-off: fixed seeds give up the per-process protection a random seed
//! gives against keys crafted to collide. The keys here come from the caller's
//! own data, not from a party the caller does not trust to a table the caller
//! exposes, and Polars' own grouping hashes with fixed seeds as well.
//!
//! Nothing observable may depend on a table's iteration order: every map keyed
//! through this hasher is sorted before it is written or reported.

use std::hash::{BuildHasherDefault, Hasher};

/// The build-hasher of [`FixedHasher`]: a `HashMap`'s or `IndexMap`'s third
/// type parameter.
pub type FixedState = BuildHasherDefault<FixedHasher>;

const MUL: u64 = 0x517c_c1b7_2722_0a95;

/// A multiply-and-rotate hasher over 64-bit words, finished with SplitMix64's
/// finalizer so that the table's low bits (which pick the bucket) and high bits
/// (which its tags use) both depend on every input bit.
///
/// Bytes are read as little-endian words, so a text hashes alike on every
/// platform.
#[derive(Default, Clone, Copy, Debug)]
pub struct FixedHasher(u64);

impl FixedHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(MUL);
    }
}

impl Hasher for FixedHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            // The length rides in the top byte, so that "a" and "a\0" differ.
            buf[7] = rest.len() as u8;
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn of<T: Hash + ?Sized>(v: &T) -> u64 {
        FixedState::default().hash_one(v)
    }

    #[test]
    fn instances_agree() {
        for k in ["", "a", "group-17", "é", "a longer key than one word"] {
            assert_eq!(of(k), of(k));
            assert_eq!(of(k), FixedState::default().hash_one(k));
        }
        // Pinned: the same in every process on every platform, which is the
        // hasher's whole purpose.
        assert_eq!(of("group-17"), 0x9c36_b920_8d67_1c14);
        assert_eq!(of(&7u64), 0x5201_ec7d_2d0b_0310);
        assert_eq!(of(&7u64), of(&7u64));
    }

    #[test]
    fn keys_that_differ_hash_apart() {
        let keys = [
            "",
            "a",
            "a\0",
            "b",
            "ab",
            "abcdefgh",
            "abcdefghi",
            "abcdefgh\0",
        ];
        let hashes: std::collections::HashSet<u64> = keys.iter().map(|k| of(*k)).collect();
        assert_eq!(hashes.len(), keys.len());
        assert_ne!(of(&Some("x")), of(&None::<&str>));
        assert_ne!(of(&1u64), of(&2u64));
    }
}
