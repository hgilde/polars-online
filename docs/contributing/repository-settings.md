<!-- kind: reference -->
# Repository settings

This page is the reference, for a contributor who maintains the GitHub
repository `hgilde/polars-online`, on every repository setting that the
release workflow and the project's security rely on: what it is set to,
and why.

**Settings are not files, so nothing in the repository restores them.**
A repository recreated under the same name gets back its workflows,
`.github/dependabot.yml` and the community health files, and none of the
settings below. Each was read on 2026-11-08 through GitHub's REST API
(`gh api repos/hgilde/polars-online` and the endpoints named in the
table), and the table gives each as it stood then.

## The settings

| setting | its value | why |
|---|---|---|
| `main: no force-push, no delete` | a ruleset on the default branch, with the `deletion` and `non_fast_forward` rules and an empty bypass list (`rulesets`) | nothing else catches a bad push: `main` cannot be rewritten or removed, and a release is committed on `main` itself |
| `release tags are immutable` | a ruleset on `refs/tags/v*`, with the `deletion` and `update` rules and an empty bypass list, so the owner is bound too (`rulesets`) | a released version's tag can never be moved or deleted. It allows *creating* a tag, which is all the release workflow does, and disabling the ruleset is the only way round it. A wrong tag is fixed by the next version, as PyPI fixes a wrong upload ([The release gate](../RELEASE-READINESS.md#the-release-gate)) |
| the `Pypi` environment | the owner as a required reviewer, with self-review allowed (`environments`) | it is the release gate: the `publish to PyPI` job declares it (`environment: pypi` in `release.yml`), and waits for the owner's approval. With a single reviewer, forbidding self-review would deadlock every release. It must exist before a release is dispatched, or the publish job fails after the wheels are built |
| the PyPI trusted publisher | on PyPI's side: `polars-online`, owner `hgilde`, repository `polars-online`, workflow `release.yml`, environment `pypi` | the upload authenticates by OIDC, so no API token is stored or rotated. PyPI keys it by those names, so a repository recreated under the same names needs nothing redone there |
| the `github-pages` environment | deployments from the branch `main` and tags `v*`, with no required reviewer (`environments`) | the Pages deploy republishes the API reference on every push to `main`, and is as reversible as the next push, so it waits for no approval |
| GitHub Pages | built by a workflow, at `https://hgilde.github.io/polars-online/` (`pages`) | the API reference; the deploy job in `ci.yml` publishes it |
| the default workflow permissions | `read`, and workflows cannot approve pull requests (`actions/permissions/workflow`) | every workflow also sets `permissions: {}` at its top and grants each job only what it uses, since a token with more scope than its job needs is the usual supply-chain foothold |
| allowed actions | all actions, with no SHA pinning required by the setting (`actions/permissions`) | every third-party action is pinned to a full commit SHA in the workflows themselves, with its version in a trailing comment, and Dependabot proposes the bumps monthly (`.github/dependabot.yml`) |
| Dependabot alerts and security updates | both on (`vulnerability-alerts`; `security_and_analysis`) | security updates open pull requests for `Cargo.lock` and `uv.lock`. Version updates come only for actions: the Rust `polars` family must move together, by hand |
| secret scanning and push protection | both on; non-provider patterns and validity checks off (`security_and_analysis`) | a credential is stopped before it reaches the public history |
| private vulnerability reporting | on (`private-vulnerability-reporting`) | it is the reporting channel `SECURITY.md` and `CODE_OF_CONDUCT.md` both point at, in place of an email address |
| CodeQL code scanning | GitHub's default setup, not a workflow in `.github/workflows/`: actions, Python and Rust, the default query suite, weekly (`code-scanning/default-setup`) | it scans the code and the workflows without a workflow to maintain |
| visibility | public | CI minutes on GitHub's hosted runners cost nothing for a public repository, and the budgets baseline is read from the newest release without a token |
| description, homepage and topics | the description "Online model fitting for Polars: linear models, streaming moments, clustering and regime detection, on data that does not fit in memory"; the homepage, the Pages site; the topics `kalman`, `online-learning`, `polars`, `python`, `regression`, `rust`, `streaming` and `time-series` | they are how anyone finds the repository |
| issues, wiki and discussions | issues on, the wiki and discussions off | issues take the templates under `.github/ISSUE_TEMPLATE/`: a bug report and a feature request |
| merging | merge commits, squash merges and rebase merges all allowed; branches are not deleted on merge | changes reach `main` by a fast-forward from a branch whose CI is green, so no merge method is relied on |

## Read the settings again

Each value in the table comes from a read-only call of the `gh` command
line, such as:

```sh
gh api repos/hgilde/polars-online --jq '{description, homepage, topics, visibility, security_and_analysis}'
gh api repos/hgilde/polars-online/rulesets
gh api repos/hgilde/polars-online/environments
```

The rulesets' rules come from `gh api repos/hgilde/polars-online/rulesets/<id>`,
and the `github-pages` environment's branches from
`gh api repos/hgilde/polars-online/environments/github-pages/deployment-branch-policies`.
