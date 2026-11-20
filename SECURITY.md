# Security policy

This repository holds only the `polars-online` and `polars-online-compat`
2.0.0 packages, each a few hundred lines of Python that forward
`import polars_online` to rheon. The library itself, and any vulnerability in it,
belongs to [rheon](https://github.com/rheon-dev/rheon).

To report a vulnerability in rheon, do **not** open a public issue. Use
the private advisory form at
<https://github.com/rheon-dev/rheon/security/advisories/new>.

A problem in the forwarding packages themselves (for example in how they
register modules or load a pickle) can be reported privately in the
**Security** tab of this repository, or at
<https://github.com/hgilde/polars-online/security/advisories/new>.

The 1.x releases of polars-online no longer receive fixes. Move to rheon
2.0.0 or later.
