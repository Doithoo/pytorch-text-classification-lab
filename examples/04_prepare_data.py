from pathlib import Path

from text_classifier.config import load_config
from text_classifier.data.manifest import prepare_data

# This example expects data/raw/ag_news_csv/{train,test}.csv.
config = load_config(Path("configs/learning_minimal.yaml"))
metadata = prepare_data(
    config["data"]["data_dir"], config["data"]["manifest_dir"], config["data"]["valid_ratio"], config["train"]["seed"]
)
print(metadata)
