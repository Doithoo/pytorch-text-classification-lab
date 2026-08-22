"""Text classification application package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-text-classification-lab")
except PackageNotFoundError:
    __version__ = "0+unknown"
