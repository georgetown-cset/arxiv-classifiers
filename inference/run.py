"""
Perform inference on a (smallish) local input file and write the results to disk.
"""
import json
import logging
import os
import sys

import typer
from srsly import read_jsonl
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)


def load_label_map(model_path):
    # Load label mapping
    with open(os.path.join(model_path, "config.json"), "rt") as f:
        config = json.load(f)
    label_names = config["label_names"]
    id_to_label = {v: k for k, v in label_names.items()}
    logger.info(f"label mapping: {id_to_label}")
    return id_to_label


def main(
    model_path,
    input_path,
    output_path,
    total: int = None,
):
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    if os.path.exists(output_path):
        raise FileExistsError(output_path)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
    )
    id_to_label = load_label_map(model_path)

    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    with open(output_path, "wt") as outfile:
        for record in tqdm(read_jsonl(input_path), total=total):
            pred = pipe(record["text"], padding=True, truncation=True)
            pred = pred[0]
            pred["label"] = id_to_label[pred["label"]]
            record.update(pred)
            outfile.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    typer.run(main)
