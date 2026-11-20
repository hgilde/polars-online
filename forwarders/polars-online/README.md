# polars-online

`polars-online` is now [`rheon`](https://pypi.org/project/rheon/). This 2.0.0
release holds no code of its own: it depends on `rheon[polars]` and
`import polars_online` forwards to it, so existing imports and saved pickles
keep working. Importing it warns once, from the line that imported it.

Install `rheon[polars]` directly, or leave `polars-online` installed and
upgrade it; either way you get rheon.

To migrate, replace `import polars_online as po` with
`from rheon import polars as rp` and call `rp.` where you called `po.`.
The `df.online` and `lf.online` namespaces are now `df.rheon` and
`lf.rheon`; `polars_online` keeps `.online` as an alias until you move.
Models, saved states and outputs are unchanged. The 1.x names that 1.x
already warned about are gone in 2.0.

`polars-online[numpy]` installs `rheon[numpy]`, the NumPy the Gram, correlation and
simulation tools need. The 1.x warning classes `PolarsOnlineWarning` and
`PolarsOnlineDeprecationWarning` are kept as names for `RheonWarning` and
`RheonDeprecationWarning`, so a warnings filter written for 1.x still matches.
`polars_online.gbdt` is `rheon.polars._gbdt`: 2.0 keeps that module out of
its public names, so it can change without a major release.
