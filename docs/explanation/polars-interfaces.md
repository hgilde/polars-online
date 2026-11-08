<!-- kind: explanation -->
# Polars interfaces

This page explains, for anyone who upgrades Polars under polars-online,
the four ways the library crosses into Polars, which of them carry a
promise, and what a break on a new Polars looks like.

**Treat a break on a new Polars as expected maintenance, because three of
the four ways this library crosses into Polars carry no promise from
Polars.** The two that stream data, `ModelBank` and the IO plugin behind
`lf.online.fit_predict`, are among the three, so check those paths first:
`ModelBank`, then the IO plugin's tests. The range of Polars the package
accepts is measured by this project, run by run, as [Versions and the
Polars pin](../VERSIONS.md#where-the-suite-has-passed) lists, and not
promised by Polars.

## Why the library carries its own Polars

**The wheel carries its own Rust `polars`, pinned exactly, and shares one
process with the py-polars you install.** Data crosses between the two
only through the four interfaces below. So the py-polars you install can
be any version in the range, and the Rust `polars` inside the wheel
does not move with it. [What is pinned](../VERSIONS.md#what-is-pinned)
gives the versions on each side.

**Two copies of Polars in one process are safe because no Rust object
crosses between them.** Data crosses on the Arrow C Data Interface, a
language-independent binary interface whose carrier, `SeriesExport`, is a
`#[repr(C)]` struct of `ArrowSchema` and `ArrowArray` pointers. No Rust
`DataFrame`, no `Drop` implementation and no ownership of a raw buffer
crosses. Each `SeriesExport` carries a `release` callback, a function
pointer into the binary that produced it, so each side frees its own
memory with its own allocator. `polars-ffi` 0.55.2's `import_series` goes
through Arrow's `import_array`. 300 round trips of 5,000-row frames, and
50 outputs kept past the inputs they came from, ran clean.

**The allocator is shared on purpose.** pyo3-polars' `PolarsAllocator`
routes every allocation this extension makes through py-polars' own
allocator, which it reaches through the capsule
`polars.polars._allocator`. Polars' own example of a compiled extension
opens with it:

```rust
use pyo3_polars::PolarsAllocator;
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
```

It keeps allocation coherent between the two copies of Polars, and it is
faster than a second, independent heap: measured on 2026-08-31, it gained
43% throughput at 5 features, 16% at 20 and 5% at 50.

**One part is fragile: the allocator's capsule.** `polars.polars` is
not a module that can be imported: it is an attribute of the `polars`
package that names the real runtime module, which on 2026-08-31 was
`_polars_runtime_32._polars_runtime`. `PyCapsule_Import` walks a dotted
name by importing and then reading attributes, which is the only reason
the lookup succeeds. And `PolarsAllocator` falls back to the system
allocator without an error when the lookup fails, so a rename in Polars
would silently cost the throughput above. `tests/test_scaffold.py`
asserts the lookup's path for that reason.

**Linking the user's Polars dynamically is not possible, because Rust has
no stable binary interface.** A Rust symbol's name carries a hash over the
compiler version, the crate's version, its features and its whole
dependency graph. py-polars' runtime exports no Polars Rust symbol to bind
to, only its Python entry points. Binding to Rust symbols would need a
compiler and a dependency graph identical to the py-polars wheel's, and
would break on every Polars patch release and every compiler update.
Rust's `dylib` crate type keeps the Rust interface, but only between
binaries built by the same compiler from the same graph, which two wheels
published months apart by different people are not. Every binding that
is resolved at run time crosses at the C level, where an interface can be
relied on: the `release` callbacks and the allocator's capsule.

## The four interfaces

**Each interface carries the promise that Polars', pyo3-polars' or
Arrow's own documentation states for it:**

| interface | used by | its promise | a break would show as |
|---|---|---|---|
| pyo3-polars' extension types | `ModelBank` | none beyond the latest definitions working with the latest Polars: provided "for convenience" | a clean `AttributeError` before any data moves |
| the IO plugin, `polars.io.plugins.register_io_source` | `lf.online.fit_predict`, `lf.online.predict`, and `with_windows` and `refresh_time` on a query | none: documented, but `@unstable` in py-polars (a warning only under `POLARS_WARN_UNSTABLE`), and its contract is partly unwritten | a wrong row count, not a crash |
| an expression's serialized form, `expr.meta.serialize(format="json")` | the window formulas, read into this library's own tree | none: Polars calls it unstable across versions; its shapes were measured the same on 1.34.0 and 1.44.2 | an unknown node, refused by name when the formula is read |
| the Arrow PyCapsule interface | [`fit_predict_arrow` and `predict_arrow`](../how-to/hand-the-output-to-arrow.md) | an Arrow specification, which py-polars and pyarrow consume. Only the output side uses it, and the frame still goes in as a Polars frame | a break of Arrow's rather than of Polars' |

## How a break shows

**In `ModelBank`, a mismatch raises an `AttributeError` before any data
moves, and does not crash the process.** `ModelBank` moves data across the boundary
through the Arrow C Data Interface, with two private methods of py-polars,
`_export` and `_import`, so a Polars without them fails at once. The
absence of `PySeries._export` is why `ModelBank` alone needs py-polars
1.28.1 or later.

**In the IO plugin, a break would show as a wrong row count, because the
unwritten part of its contract is its pushdowns.** Polars pushes a
projection, a predicate and a slice into a Python source, and does not
re-apply any of them to what the source returns. So the source honors all
three itself (`python/polars_online/_frame.py`, and `stream.py` for
`with_windows` and `refresh_time`). `tests/test_frame.py` checks every
pushdown against the collected frame, in both orders, and the weekly run
on the newest Polars runs those tests.

**The serialized form is read narrowly, because it is not stable.** Seven
node kinds are read, when the formula is taken, under the caller's own
Polars, and an unknown node is refused by name. Python and Rust then
rebuild the formula through Polars' public builders
(`python/polars_online/_formula.py`).
