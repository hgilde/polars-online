"""Warn, before the compiled module loads, that the CPU lacks a feature the wheel needs.

The x86-64 wheels of ``rheon`` are compiled for CPUs with AVX2, FMA
and the rest of the feature set in ``BUILD_FEATURE_FLAGS``; the compiler may
use any of them anywhere. On a CPU without one the first such instruction
kills the process with an illegal-instruction signal, and no Python frame
says why. This module reads the CPU's features with the ``cpuid``
instruction and, when one is missing, issues a ``RuntimeWarning`` that names
it and the package to install instead (``rheon-compat``, the same
code built for every x86-64 CPU). It runs in pure Python, first in the
package's ``__init__``: a check inside the extension would run after the
compiler had already emitted instructions it cannot execute.

The check does nothing, and loads nothing, where there is nothing to check:
on ARM, in ``rheon-compat``, in a build from source with no
``target-feature`` flag (all of which record an empty ``BUILD_FEATURE_FLAGS``),
and when ``RHEON_SKIP_CPU_CHECK`` is set. A CPU it cannot read
(an operating system that refuses executable memory, say) is not a reason to
fail the import: it reads nothing and says nothing.

The design, the 14 feature names and the machine code are Polars'
(``polars/_cpu_check.py``, MIT), which vendors Anders Høst's ``cpuid.py``
(MIT, Copyright (c) 2014 Anders Høst): a few bytes that run ``cpuid`` from
Python through ``ctypes``. The bytes here were assembled again from the
instructions listed beside them, and the code around them is written
afresh, with two additions: a leaf is read only when the CPU reports it, and
a failure to read is silent.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import sys
import warnings
from collections.abc import Callable

from rheon import _env

#: Set it to anything non-empty to skip the check.
SKIP_ENV = "RHEON_SKIP_CPU_CHECK"

#: The other package, built for every x86-64 CPU.
COMPAT_PACKAGE = "rheon-compat"

#: Each feature the check can read: the ``cpuid`` leaf, the register and the
#: bit, from the Intel SDM (vol. 2A, ``CPUID``) and AMD's APM (vol. 3).
#: The names are the ones rustc's ``-C target-feature`` takes.
FEATURE_BITS: dict[str, tuple[int, str, int]] = {
    "sse3": (1, "ecx", 0),
    "pclmulqdq": (1, "ecx", 1),
    "ssse3": (1, "ecx", 9),
    "fma": (1, "ecx", 12),
    "cmpxchg16b": (1, "ecx", 13),
    "sse4.1": (1, "ecx", 19),
    "sse4.2": (1, "ecx", 20),
    "movbe": (1, "ecx", 22),
    "popcnt": (1, "ecx", 23),
    "avx": (1, "ecx", 28),
    "bmi1": (7, "ebx", 3),
    "avx2": (7, "ebx", 5),
    "bmi2": (7, "ebx", 8),
    "lzcnt": (0x80000001, "ecx", 5),
}

_X86_MACHINES = {"x86_64", "amd64", "x64"}

# ``void cpuid(uint32_t out[4], uint32_t leaf, uint32_t subleaf)``: leaf in
# eax, subleaf in ecx, and eax, ebx, ecx, edx stored to ``out``. System V
# (Linux, macOS): out, leaf, subleaf arrive in rdi, esi, edx.
# fmt: off
_POSIX_CODE = bytes([
    0x53,                    # push   %rbx
    0x89, 0xf0,              # mov    %esi,%eax
    0x89, 0xd1,              # mov    %edx,%ecx
    0x0f, 0xa2,              # cpuid
    0x89, 0x07,              # mov    %eax,(%rdi)
    0x89, 0x5f, 0x04,        # mov    %ebx,0x4(%rdi)
    0x89, 0x4f, 0x08,        # mov    %ecx,0x8(%rdi)
    0x89, 0x57, 0x0c,        # mov    %edx,0xc(%rdi)
    0x5b,                    # pop    %rbx
    0xc3,                    # ret
])
# Windows x64: out, leaf, subleaf arrive in rcx, edx, r8d.
_WINDOWS_CODE = bytes([
    0x53,                    # push   %rbx
    0x89, 0xd0,              # mov    %edx,%eax
    0x49, 0x89, 0xc9,        # mov    %rcx,%r9
    0x44, 0x89, 0xc1,        # mov    %r8d,%ecx
    0x0f, 0xa2,              # cpuid
    0x41, 0x89, 0x01,        # mov    %eax,(%r9)
    0x41, 0x89, 0x59, 0x04,  # mov    %ebx,0x4(%r9)
    0x41, 0x89, 0x49, 0x08,  # mov    %ecx,0x8(%r9)
    0x41, 0x89, 0x51, 0x0c,  # mov    %edx,0xc(%r9)
    0x5b,                    # pop    %rbx
    0xc3,                    # ret
])
# fmt: on


def baked_flags() -> str:
    """The features this build was compiled to need, as ``"+avx2,+fma,..."``.

    Empty when the build recorded none, or recorded nothing at all (no
    ``_build_flags`` module: a checkout that was never built through cargo).
    """
    try:
        module = importlib.import_module("rheon._build_flags")
    except ImportError:
        return ""
    return str(getattr(module, "BUILD_FEATURE_FLAGS", ""))


def required_features(flags: str) -> list[str]:
    """The feature names in a flag string, without their ``+``; a feature
    switched off (``-name``) and an empty item are not required."""
    return [f.lstrip("+") for f in flags.split(",") if f and not f.startswith("-")]


def _is_x86_64() -> bool:
    return platform.machine().lower() in _X86_MACHINES and sys.maxsize > 2**32


class _Cpuid:
    """``cpuid`` as a Python callable: a few bytes in executable memory."""

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        windows = os.name == "nt"
        code = _WINDOWS_CODE if windows else _POSIX_CODE
        self._release: Callable[[], object] | None = None
        if windows:
            kernel32 = ctypes.CDLL("kernel32.dll")
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            kernel32.VirtualFree.restype = ctypes.c_long
            kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
            mem_commit_reserve, page_execute_readwrite, mem_release = 0x3000, 0x40, 0x8000
            address = kernel32.VirtualAlloc(
                None, len(code), mem_commit_reserve, page_execute_readwrite
            )
            if not address:
                raise OSError("no memory for the CPU check")
            ctypes.memmove(address, code, len(code))
            self._release = lambda: kernel32.VirtualFree(address, 0, mem_release)
        else:
            import mmap

            # Written first and made executable after, since some systems
            # refuse memory that is writable and executable at once.
            self._map = mmap.mmap(
                -1,
                len(code),
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self._map.write(code)
            address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
            mprotect = ctypes.CDLL(None, use_errno=True).mprotect
            mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            mprotect.restype = ctypes.c_int
            if mprotect(address, len(code), mmap.PROT_READ | mmap.PROT_EXEC) != 0:
                raise OSError("could not make memory executable for the CPU check")
            self._release = self._close_map
        prototype = ctypes.CFUNCTYPE(
            None, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_uint32
        )
        self._function = prototype(address)

    def _close_map(self) -> None:
        # A view still alive is left to the garbage collector.
        with contextlib.suppress(BufferError):
            self._map.close()

    def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
        out = (self._ctypes.c_uint32 * 4)()
        self._function(out, leaf, subleaf)
        return dict(zip(("eax", "ebx", "ecx", "edx"), out, strict=True))

    def close(self) -> None:
        if self._release is not None:
            self._release()
            self._release = None


def read_cpu_features() -> dict[str, bool] | None:
    """Which of ``FEATURE_BITS`` this CPU has, or ``None`` when it cannot be
    read: not an x86-64 CPU, or memory it may not execute."""
    if not _is_x86_64():
        return None
    try:
        cpuid = _Cpuid()
    except Exception:
        return None
    try:
        registers: dict[int, dict[str, int]] = {}
        highest = {0: cpuid(0)["eax"], 0x80000000: cpuid(0x80000000)["eax"]}
        for leaf in sorted({leaf for leaf, _, _ in FEATURE_BITS.values()}):
            # A leaf past the highest the CPU reports returns the highest
            # one's data, which is no answer: its features are absent.
            top = highest[0x80000000 if leaf >= 0x80000000 else 0]
            registers[leaf] = cpuid(leaf) if leaf <= top else {}
    except Exception:
        return None
    finally:
        cpuid.close()
    return {
        name: bool(registers[leaf].get(register, 0) & (1 << bit))
        for name, (leaf, register, bit) in FEATURE_BITS.items()
    }


def missing_features(flags: str | None = None) -> list[str]:
    """The features of ``flags`` (this build's, by default) the CPU lacks, in
    the order the build names them. A feature the check cannot read is not
    reported, and neither is anything when the CPU cannot be read."""
    wanted = required_features(baked_flags() if flags is None else flags)
    if not wanted:
        return []
    have = read_cpu_features()
    if have is None:
        return []
    return [f for f in wanted if f in have and not have[f]]


def message(missing: list[str]) -> str:
    """The warning's text."""
    return (
        "Missing required CPU features.\n"
        "\n"
        "The following CPU features, which this build of rheon needs, were not "
        "detected:\n"
        f"    {', '.join(missing)}\n"
        "Continuing to use this build on this processor will likely crash it with an "
        "illegal instruction.\n"
        f"Install `{COMPAT_PACKAGE}` instead of `rheon`: it is the same code, built "
        "for every x86-64 CPU, and it is imported as `rheon` too. Install one or the "
        "other: run `pip uninstall rheon` first.\n"
        "\n"
        "Hint: If you are on an Apple silicon Mac (M1 or later), this is likely due to "
        "running Python under Rosetta. Install a native Python that does not run under "
        "Rosetta x86-64 emulation.\n"
        "\n"
        "If you believe this warning to be a false positive, set the "
        f"`{SKIP_ENV}` environment variable to bypass this check.\n"
    )


def run() -> None:
    """The check: warn if this CPU lacks a feature the build needs."""
    if _env.get(SKIP_ENV):
        return
    missing = missing_features()
    if missing:
        warnings.warn(message(missing), RuntimeWarning, stacklevel=3)
