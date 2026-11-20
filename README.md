# polars-online is now rheon

From 2.0.0, polars-online is called **rheon**. It lives at
[github.com/rheon-dev/rheon](https://github.com/rheon-dev/rheon), and its
documentation is there. To install it:

```
pip install "rheon[polars]"
```

## The forwarding packages

`polars-online` 2.0.0 and `polars-online-compat` 2.0.0 contain no models.
They are small packages that depend on `rheon[polars]` and `rheon-compat[polars]`,
so that code written for 1.x keeps running while you move.

`import polars_online` does two things:

- It warns once, with a `DeprecationWarning`, that polars-online is now rheon,
  and points at the line that imported it.
- It then gives you rheon's Polars interface under the old names. Every
  `polars_online.<module>` is the same module as its rheon counterpart, and the
  `.online` namespace on a DataFrame or LazyFrame is the same as `.rheon`.

A pickle saved by polars-online 1.x loads and carries on with either package
installed. The models, their saved states and their outputs are unchanged.

To move off the forwarder, replace `import polars_online as po` with
`from rheon import polars as rp`, call `rp.` where you called `po.`, and write
`df.rheon` where you wrote `df.online`.

## The 1.x line

The last 1.x release was 1.6.0. Its source stays in this repository's history, at the
tag [v1.6.0](https://github.com/hgilde/polars-online/tree/v1.6.0).
