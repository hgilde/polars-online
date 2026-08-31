## What this changes

<!-- One paragraph. If it fixes a defect, say what the defect was and how it
     could be observed -- that is what the commit history here looks like. -->

## Checklist

- [ ] `./scripts/gate.sh` passes (run unpiped; it ends with `gate: PASS`)
- [ ] Golden numbers unchanged — or, if they moved, the PR explains why that is
      correct rather than regenerated
- [ ] New behaviour has a test with an **oracle** (a longhand recursion, an
      equivalent configuration, or the optimality conditions), not just a
      pinned output
- [ ] Relevant doc updated in the same commit (`docs/PLAN.md`,
      `docs/ENHANCEMENTS.md`, `docs/TESTING.md`, `docs/PERFORMANCE.md`)

## Performance

<!-- Only if this touches the hot path. Numbers before and after, from the
     scripts in docs/PERFORMANCE.md, measured on an idle machine. -->
