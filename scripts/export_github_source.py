"""Export only reviewed source categories, without Git history or local artifacts."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    ".gitattributes", ".gitignore", "CITATION.cff", "CONTRIBUTING.md",
    "LICENSE", "MANIFEST.in", "README.md", "pyproject.toml",
)
TREES = {
    ".github": {".yml", ".md"},
    "docs": {".md"},
    "scripts": {".py"},
    "src": {".py", ".json"},
    "tests": {".py"},
}


def files():
    selected = [ROOT / name for name in TOP_LEVEL]
    for folder, suffixes in TREES.items():
        for path in (ROOT / folder).rglob("*"):
            parts = path.relative_to(ROOT / folder).parts
            if path.is_file() and path.suffix in suffixes and not any(
                part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info")
                for part in parts
            ):
                selected.append(path)
    for path in sorted(selected):
        if not path.is_file() or path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Missing file or unexpected symlink: {path}")
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/github-source.zip")
    args = parser.parse_args()
    selected = list(files())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(ROOT))
    print(f"Exported {len(selected)} source files to {args.output.resolve()}")


if __name__ == "__main__":
    main()
