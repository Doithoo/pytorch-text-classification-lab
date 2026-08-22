from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote

from text_classifier.cli import build_parser
from text_classifier.models import list_models

ROOT = Path(__file__).parents[1]
PUBLICATION_ROOTS = [ROOT / name for name in ("docs", "configs", "examples", "scripts", "tests", "src")]
ROOT_PAIRS = ["README", "CONTRIBUTING"]
WORKFLOW = "download -> prepare -> inspect -> dry run -> train -> evaluate -> predict"


def _publication_pages() -> list[Path]:
    pages = [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "CONTRIBUTING.md", ROOT / "CONTRIBUTING.zh-CN.md"]
    for directory in PUBLICATION_ROOTS:
        pages.extend(directory.rglob("*.md"))
    return sorted(set(pages))


def _broken_local_links(pages: list[Path]) -> list[str]:
    missing = []
    for source in pages:
        for raw_target in re.findall(r"!?\[[^]]*]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            target = unquote(raw_target.split()[0]).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (source.parent / target).resolve().exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    return missing


def test_english_and_chinese_pages_exist_in_pairs() -> None:
    missing = []
    for stem in ROOT_PAIRS:
        for path in (ROOT / f"{stem}.md", ROOT / f"{stem}.zh-CN.md"):
            if not path.is_file():
                missing.append(path.relative_to(ROOT).as_posix())
    for directory in PUBLICATION_ROOTS:
        for chinese in directory.rglob("*.zh-CN.md"):
            english = chinese.with_name(chinese.name.replace(".zh-CN.md", ".md"))
            if not english.is_file():
                missing.append(english.relative_to(ROOT).as_posix())
        for english in directory.rglob("*.md"):
            if english.name.endswith(".zh-CN.md"):
                continue
            chinese = english.with_name(english.name.removesuffix(".md") + ".zh-CN.md")
            if not chinese.is_file():
                missing.append(chinese.relative_to(ROOT).as_posix())
    assert not missing, "missing language page(s):\n" + "\n".join(sorted(set(missing)))


def test_all_local_markdown_links_resolve() -> None:
    missing = _broken_local_links(_publication_pages())
    assert not missing, "broken local links:\n" + "\n".join(missing)


def test_readmes_publish_workflow_and_recorded_metric() -> None:
    for path in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        content = path.read_text(encoding="utf-8")
        assert WORKFLOW in content
        assert "0.914610" in content
        assert "docs/recorded-run/kaggle-agnews-textcnn/" in content
        assert "prepare-data --data-dir" not in content


def test_recorded_run_index_is_current() -> None:
    for path in (ROOT / "docs/recorded-run/README.md", ROOT / "docs/recorded-run/README.zh-CN.md"):
        content = path.read_text(encoding="utf-8")
        assert "0.914610" in content
        assert "does not yet" not in content
        assert "目前不声明" not in content


def test_documented_cli_lines_use_real_parser_options() -> None:
    parser = build_parser()
    failures = []
    for source in _publication_pages():
        for line in source.read_text(encoding="utf-8").splitlines():
            command = line.strip()
            if command.endswith("\\"):
                continue
            if command.startswith("uv run text-classify "):
                command = command.removeprefix("uv run ")
            elif not command.startswith("text-classify "):
                continue
            try:
                parser.parse_args(shlex.split(command)[1:])
            except SystemExit as exc:
                if exc.code != 0:
                    failures.append(f"{source.relative_to(ROOT)}: {command}")
    assert not failures, "invalid documented CLI command(s):\n" + "\n".join(failures)


def test_documented_python_entry_points_and_configs_exist() -> None:
    missing = []
    for source in _publication_pages():
        content = source.read_text(encoding="utf-8")
        for target in re.findall(r"uv run(?: --extra \w+)? python\s+([^\s`\\]+)", content):
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
        for target in re.findall(r"--config\s+([^\s`\\<]+)", content):
            if target == "PATH":
                continue
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "missing documented path(s):\n" + "\n".join(missing)


def test_model_catalog_lists_every_registered_model() -> None:
    for suffix in ("", ".zh-CN"):
        content = (ROOT / f"docs/reference/model-zoo{suffix}.md").read_text(encoding="utf-8")
        documented = {name for name in list_models() if re.search(rf"\|\s*`{re.escape(name)}`\s*\|", content)}
        assert documented == set(list_models())


def test_generated_documentation_assets_are_nonempty_pngs() -> None:
    for name in ("ag-news-textcnn-training.png", "ag-news-textcnn-confusion.png"):
        path = ROOT / "docs/assets" / name
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000
