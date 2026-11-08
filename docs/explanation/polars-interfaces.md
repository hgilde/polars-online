<!-- kind: explanation -->
# Polars interfaces

This page explains, for anyone who upgrades Polars under polars-online,
the four ways the library crosses into Polars, which of them carry a
promise, and what a break on a new Polars looks like.

**Treat a break on a new Polars as expected maintenance, because three of
the four ways this library crosses into Polars carry no promise from
Polars.** The two that stream data, `ModelBank` and the IO plugin behind
`lf.online.fit_predict`, are among the three, so check those paths first.
The range of Polars the package accepts is measured by this project, run
by run, as [Versions and the Polars pin](../VERSIONS.md#where-the-suite-has-passed)
lists, and not promised by Polars.

## Why the library carries its own Polars

**The wheel carries its own Rust `polars`, pinned exactly, and shares one
process with the py-polars you install.** Data crosses between the two
only through the four interfaces below. So the py-polars you install can
be any version in the range, and the Rust `polars` inside the wheel
does not move with it. [What is pinned](../VERSIONS.md#what-is-pinned)
gives the versions on each side.

## The four interfaces

**Each interface carries the promise that Polars', pyo3-polars' or
Arrow's own documentation states for it:**

| interface | used by | its promise |
|---|---|---|
| pyo3-polars' extension types | `ModelBank` | none beyond the latest definitions working with the latest Polars: provided "for convenience" |
| the IO plugin | `lf.online.fit_predict`, `lf.online.predict`, and `with_windows` and `refresh_time` on a query | none: documented, but `@unstable` in py-polars |
| an expression's serialized form, `expr.meta.serialize(format="json")` | the window formulas, read into this library's own tree | none: Polars calls it unstable across versions; its shapes were measured the same on 1.34.0 and 1.44.2 |
| the Arrow PyCapsule interface | [`fit_predict_arrow` and `predict_arrow`](../how-to/hand-the-output-to-arrow.md) | an Arrow specification, which py-polars and pyarrow consume. Only the output side uses it, and the frame still goes in as a Polars frame |

## How a break shows

**In `ModelBank`, a mismatch raises an `AttributeError` before any data
moves, and does not crash the process.** `ModelBank` moves data across the boundary
through the Arrow C Data Interface, with two private methods of py-polars,
`_export` and `_import`, so a Polars without them fails at once.
[Which interfaces carry a
promise](../RELEASE-READINESS.md#which-interfaces-carry-a-promise) in
the release guide says how a break shows on each interface, and which
tests watch for it.
