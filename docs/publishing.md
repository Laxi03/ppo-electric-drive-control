# Publishing to GitHub

This is a Python package, not a web service. The repository's [GitHub Actions workflow](../.github/workflows/ci.yml) builds a source distribution and wheel, installs the wheel, checks dependencies, and runs tests on Python 3.12. It does not publish packages, deploy servers, or upload models.

## First public push

1. Run the tests and review the source, license, and citation metadata.
2. Review third-party attribution and confirm rights to publish the source. Do not include vendor datasheets, thesis drafts, original measurement dumps, local paths, credentials, or checkpoint binaries.
3. If the existing repository has a private or unreviewed history, use a clean public source snapshot. Removing a file from the current tree does not remove it from earlier commits, and Git ignores do not remove tracked files.
4. Push the reviewed source to the chosen GitHub repository. Check the Actions run and the rendered README before tagging a release.

## Source-only snapshot

From the project directory:

```sh
python scripts/export_github_source.py
```

This creates `dist/github-source.zip` from explicit source directories and metadata files. It excludes Git history, the existing index, models, archives, generated reports, and virtual environments. Unpack it in a **new empty directory** for a fresh public repository, and review the file list before staging. The export is not a substitute for reviewing the contents for secrets, third-party material, or other data that should not be released. Use `--output` to export to a different filename.

For a distribution build, use `python -m pip install build` and `python -m build`. Publish a GitHub release only after reviewing CI; provide provenance, compatibility, and licensing separately for any future checkpoint release.
