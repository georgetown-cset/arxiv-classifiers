import csv
import os
import shutil
from pathlib import Path

import torch
import typer
from transformers import AutoConfig, AutoModelForSequenceClassification


def convert(model_dir: Path):
    assert model_dir.is_dir()
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    torch.save(model.state_dict(), model_dir / "model.pth")


def main():
    with open("models.csv", "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_dir = Path("../models", row["model_name"])
            checkpoint_dir = model_dir / f"checkpoint-{row['checkpoint_number']}"
            image_dir = Path("../images", row["image_name"])
            image_model_dir = image_dir / "model"
            image_model_dir.mkdir(exist_ok=True, parents=True)
            # Copy e.g. ../models/ai-32/config.json to ../images/ai-32/config.json
            shutil.copy(model_dir / "config.json", image_dir)
            # From e.g. ../models/ai-32/checkpoint-68354/pytorch_model.bin ...
            #   create ../models/ai-32/checkpoint-68354/model.pth
            convert(checkpoint_dir)
            # Copy model.pth to e.g. ../images/ai-32/model/model.pth
            shutil.copy(checkpoint_dir / "model.pth", image_model_dir)


if __name__ == "__main__":
    typer.run(main)
