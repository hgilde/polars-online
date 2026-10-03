## What this changes

<!-- One paragraph. If it fixes a defect, say what the defect was and how it
     could be observed -- that is what the commit history here looks like. -->

## Checklist

- [ ] `./scripts/gate.sh` passes (run unpiped; it ends with `gate: PASS`)
- [ ] Golden numbers unchanged — or, if they moved, the PR explains why that is
      correct rather than regenerated
- [ ] New behaviour has a test with an **oracle**, beyond a pinned output.
      That is another library that computes the same quantity where one
      does, or else a longhand recursion, an equivalent configuration, or
      the optimality conditions
- [ ] Relevant doc updated in the same commit (`docs/PLAN.md`,
      `docs/ENHANCEMENTS.md`, the README,
      `CHANGELOG.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`,
      `docs/OUTPUTS.md` regenerated), written to `docs/WRITING.md`
- [ ] A public API change: the diff of `tests/api_surface.txt` is in this PR

## Performance

<!-- Only if this touches the hot path. Numbers before and after, from the
     scripts in docs/PERFORMANCE.md, measured on an idle machine. -->
