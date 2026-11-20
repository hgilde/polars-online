# polars-online

From 2.0.0 this repository only ships two thin forwarding packages,
`polars-online` and `polars-online-compat` (`forwarders/`). Each warns once on
`import polars_online` and forwards to `rheon` and `rheon-compat`. The library
is rheon, at https://github.com/rheon-dev/rheon; its 1.x source is the tag `v1.6.0`
of this repository.

## Commands

```
python -m pytest forwarders/tests -q                 # stub mode: no network, no wheels
FORWARDER_REAL_PYTHON=<env with rheon[polars,numpy]==2.0.0>/bin/python \
FORWARDER_OLD_PYTHON=<env with polars-online[numpy]==1.6.0>/bin/python \
    python -m pytest forwarders/tests -q             # real mode (rheon-compat and
                                                     # polars-online-compat work the same way)
(cd forwarders/polars-online && uv build)            # likewise polars-online-compat
python scripts/verify_forwarder.py --wheel <built wheel>
```

## Release

`.github/workflows/release.yml` is dispatched on `main`, never tagged by hand. With
`publish` off it builds and tests (a rehearsal). With `publish` on, it uploads after
the `test` job passes, verifies from PyPI, and tags `v2.0.0`. **The user dispatches it;
approving the `pypi` environment is the user's click.** Publishing is trusted
publishing (no token); nothing else in the repository uploads.
