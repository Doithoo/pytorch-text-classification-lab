from pathlib import Path

from text_classifier.data.manifest import manifest_identity

manifest_dir = Path("data/manifests")
print("manifest identity:", manifest_identity(manifest_dir))
print("Use text-classify evaluate and predict for checkpoint inference.")
