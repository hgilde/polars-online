//! A cache line of stack under `run_instance`, moved out of `stream.rs`,
//! which is held to the size a source file may reach (task 377).

/// 64 bytes aligned to 64, never written: a local of this type, its address
/// taken ([`Self::keep`]), makes the function that holds it align its frame
/// to a cache line, so every frame below it -- the model's step, its solve,
/// faer's calls -- falls at the same place within a line whatever the size
/// of the frames above.
///
/// `run_instance`'s frame changes size with most edits to it, and where a
/// model's frames fell within a line moved with it. Task 375 grew it by 32
/// bytes, and `ewridge` at 20 features, solving every few rows, ran 3-6%
/// slower on arm64 in both function layouts, running the same instructions
/// at the same addresses: the time went to the first loads of faer's
/// product (`matmul_imp`, 3.8 times its samples), which read the views its
/// caller had just stored on the stack, now 32 bytes on. A local of 32
/// bytes more gave the speed back, as this does at any frame size.
#[repr(align(64))]
pub(super) struct StackLine(std::mem::MaybeUninit<[u8; 64]>);

impl StackLine {
    #[inline(always)]
    pub(super) fn new() -> Self {
        Self(std::mem::MaybeUninit::uninit())
    }

    /// Takes the line's address, so that it is kept on the stack, aligned,
    /// until here.
    #[inline(always)]
    pub(super) fn keep(&self) {
        std::hint::black_box(&self.0);
    }
}
