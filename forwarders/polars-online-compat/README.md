# polars-online-compat

`polars-online-compat` is now
[`rheon-compat`](https://pypi.org/project/rheon-compat/), the build of rheon for
x86-64 CPUs without AVX2 and FMA. This 2.0.0 release holds no code of its own:
it depends on `rheon-compat[polars]` and `import polars_online` forwards to
it, so existing imports and saved pickles keep working. Importing it warns
once.

Install `rheon-compat[polars]` directly, or leave `polars-online-compat`
installed and upgrade it. It is imported as `polars_online`, like
`polars-online`, so install one or the other.

To migrate, replace `import polars_online as po` with
`from rheon import polars as rp` and call `rp.` where you called `po.`.
The `df.online` and `lf.online` namespaces are now `df.rheon` and
`lf.rheon`; `polars_online` keeps `.online` as an alias until you move.
