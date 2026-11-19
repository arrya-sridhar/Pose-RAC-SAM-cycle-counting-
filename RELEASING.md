# Versions and releases

How this project numbers its versions, keeps them apart, and publishes them.

## Version numbers

| Kind | Example | When |
| --- | --- | --- |
| New whole number | v5 → v6 → v7 | The way the model works changes (tracking, counting or timing method) and results change |
| Point version | v6 → v6.1 → v6.2 | An improvement or extension within the same method that changes results (new settings, a new setup step, new pre-processing) |
| Patch | v6.1 → v6.1.1 | Results stay exactly the same: clean-up, documentation, packaging, or a bug fix that does not change outputs (shown by the regression check) |
| Experiment | v5.1 | A variant that was tested but not adopted: an entry in CHANGELOG.md and HISTORY.md, but no tag and no release |

v1–v4 were made before the project was in git; their code is kept in `history/` and they have no tags.

## Branches and tags

- `main` always holds the latest released version.
- Work on a new version happens on its own branch, named after it: `v6.2-dev`, `v7-dev`, and so on.
- A version is merged into `main` only when it is finished, documented, and has passed the
  [regression check](README.md#regression-check) (or the release notes say why it was not run).
- Each release is an annotated git tag on `main`, named exactly as the version (`v6.1`), whose message is a one-line
  summary plus the headline results.
- A published tag is never moved or deleted. A mistake found later is fixed in a new patch version.

## What every release contains

1. **An entry in [`CHANGELOG.md`](CHANGELOG.md):** date, what changed, headline results, where the details are.
2. **An entry in [`HISTORY.md`](HISTORY.md) Section 10**, using the template there (why, how general, settings frozen
   before testing, regression check, results, side effects, decision).
3. **The README updated:** the "current version" line, the results, and anything about how to run it.
4. **Release notes in `docs/releases/<version>.md`**, with these sections: Summary, What changed, Results (per test
   set), How to run, Known limits, Files.
5. **A GitHub Release** for the tag, with the release notes as its text and these files attached when they exist:
   reports (PDF), result tables, demo videos.
   Not attached: model weights (the README says where to download them) and datasets.
6. A version that is still being tested can be published as a GitHub *pre-release*.

## How to make a release

```bash
# on main, after merging the finished version branch
git tag -a v6.1 -m "v6.1: <one-line summary>. <headline results>"
git push origin main v6.1
gh release create v6.1 --verify-tag --title "v6.1: <one-line summary>" \
   --notes-file docs/releases/v6.1.md  <files to attach>
```
