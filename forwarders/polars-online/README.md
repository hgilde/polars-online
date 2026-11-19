# polars-online

`polars-online` is now [`rheon`](https://pypi.org/project/rheon/). This 2.0.0
release holds no code of its own: it depends on `rheon[polars]` and
`import polars_online` forwards to it, so existing imports and saved pickles
keep working. Importing it warns once.

Install `rheon[polars]` directly, or leave `polars-online` installed and
upgrade it; either way you get rheon.

To migrate, replace `import polars_online as po` with
`from rheon import polars as rp` and call `rp.` where you called `po.`.
The `df.online` and `lf.online` namespaces are now `df.rheon` and
`lf.rheon`; `polars_online` keeps `.online` as an alias until you move.
Models, saved states and outputs are unchanged. The 1.x names that 1.x
already warned about are gone in 2.0.
