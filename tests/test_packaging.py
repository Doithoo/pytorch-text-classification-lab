from importlib.metadata import metadata, version
from pathlib import Path

import yaml

from text_classifier import __version__

ROOT = Path(__file__).parents[1]


def test_installed_version_has_one_metadata_source() -> None:
    assert __version__ == version("pytorch-text-classification-lab")
    assert __version__ == "0.3.0"


def test_package_metadata_has_project_links_and_license() -> None:
    package = metadata("pytorch-text-classification-lab")
    assert package["License-Expression"] == "MIT"
    homepage = package.get("Home-page", "")
    project_urls = package.get_all("Project-URL", [])
    assert "pytorch-text-classification-lab" in homepage or any(
        "pytorch-text-classification-lab" in value for value in project_urls
    )


def test_manifest_lists_public_configs_and_readmes() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "README.zh-CN.md" in manifest
    assert "recursive-include configs *.yaml" in manifest
    for config in (ROOT / "configs").glob("*.yaml"):
        assert isinstance(yaml.safe_load(config.read_text(encoding="utf-8")), dict)
