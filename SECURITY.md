# Security policy

## Supported versions

This project is pre-1.0. Only the latest released version receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private reporting: the **Security** tab of this repository →
**Report a vulnerability**. That opens a private advisory visible only to the
maintainers.

Expect an acknowledgement within a week. If a fix is warranted, it ships as a
patch release with the advisory published alongside it.

## Scope

This is a numerical library with no network access, no deserialization of
untrusted formats other than its own state files, and no `unsafe` code in
`online-core` (enforced by `unsafe_code = "forbid"`).

The parts worth scrutiny:

- **State files** (`ModelBank.save` / `load`) are versioned msgpack. Loading a
  state file from an untrusted source is loading untrusted input; the loader
  validates a magic string, a format version and a schema version, but it is
  not a hardened parser. Treat state files like pickles: load only ones you
  produced.
- **The CLI's TOML config** names input and output paths, which the process
  then reads and writes.
