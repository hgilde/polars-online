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

Two kinds of file are parsed by this library rather than by Polars: its own
state files, and the command line's TOML configuration. The command line
reads its input data through Polars' readers, as parquet, IPC, CSV or NDJSON.

The parts worth scrutiny:

- **State files** (`ModelBank.save` / `load`) are versioned msgpack. One
  loader reads them, whether a file arrives through `ModelBank.load` or
  `ModelBank.load_bytes`, through `load_state=` on a query, or through the
  command line's `--resume` or its TOML's `load_state`. Loading a state file
  from an untrusted source is loading untrusted input. The loader validates
  a magic string, a format version and a schema version, but it is not a
  hardened parser. Treat state files like pickles: load only ones you
  produced.
- **The CLI's TOML config** names input and output paths, which the process
  then reads and writes.
