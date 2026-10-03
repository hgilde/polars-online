# Security policy

## Supported versions

This project is pre-1.0. Only the latest released version receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private reporting: the **Security** tab of this repository →
**Report a vulnerability**, or go straight to
<https://github.com/hgilde/polars-online/security/advisories/new>. That
opens a private advisory visible only to the maintainers.

Expect an acknowledgement within a week. If a fix is warranted, it ships as a
patch release with the advisory published alongside it.

## Scope

This is a numerical library with no network access. There is no `unsafe`
code in `online-core`, enforced by `unsafe_code = "forbid"`.

This library parses two kinds of file itself, rather than through Polars:
its own state files, and the command line's TOML configuration. The command
line reads its input data through Polars' readers, as parquet, IPC, CSV or
NDJSON.

**Treat a state file like a pickle: load only ones you produced.** Each
loader checks a magic string and a version, but none is a hardened parser.
There are three state formats, each versioned msgpack:

| state | written by | read by |
|---|---|---|
| a bank's | `ModelBank.save` and `save_bytes`, `save_state=` on a query, the command line's `--save-state` | `ModelBank.load` and `load_bytes`, `load_state=` on a query, the command line's `--resume` or its TOML's `load_state` |
| a window run's | `po.stream.with_windows(save_state=)` | `po.stream.with_windows(load_state=)`, also inside a query |
| a refresh-time grid's | `po.stream.refresh_time(save_state=)` | `po.stream.refresh_time(load_state=)` |

A window run's state carries the rows it holds as Arrow IPC, which Polars'
IPC reader reads back. A bank file with a window target embeds one window
state per group, and the expression the target is built from. That
expression is rebuilt from a fixed set of element-wise node kinds, and any
other node is refused.

**The CLI's TOML config** names input and output paths, which the process
then reads and writes.
