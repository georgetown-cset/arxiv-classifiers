import argparse
import json
import os
import sys
from typing import Iterator

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.ml.inference.base import KeyedModelHandler, RunInference
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor, make_tensor_model_fn
from apache_beam.options.pipeline_options import PipelineOptions
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification


class Tokenize(beam.DoFn):
    def __init__(self, tokenizer: AutoTokenizer, id_field: str = "id"):
        self._tokenizer = tokenizer
        self.id_field = id_field

    def process(self, element):
        """Process the raw text input to a format suitable for model inference.

            Args:
              element: A string of text

            Returns:
              A tokenized example.
            """
        text = element.get("text", "")
        input_ids = self._tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512).input_ids
        return [(element[self.id_field], input_ids.squeeze())]


class Postprocess(beam.DoFn):

    def __init__(self, id_field: str = "id"):
        self.id_field = id_field

    def process(self, element):
        """
            Extract predictions and predicted probabilities from the PredictionResult.

            Args:
              element: The RunInference output to be processed.
            """
        import torch.nn.functional as nnf
        # The key here is a document ID; in the case of arXiv data something like "1010.0220"
        # The result is an apache_beam.ml.inference.base.PredictionResult
        # It has a tensor attribute with the input_ids, and an inference attribute containing a "logits" tensor
        key, result = element
        # result.inference is a dict like {'logits': tensor([ 4.4492, -3.5480])}
        # probs is a tensor like tensor([9.9966e-01, 3.3631e-04])
        probs = nnf.softmax(result.inference["logits"], dim=0)
        # argmax over the probs tensor gives us the index of the higher probability
        prediction = probs.argmax()
        # breakpoint()
        yield {
            self.id_field: key,
            "prediction": int(prediction),
            "probability": float(probs[prediction])
        }


class PytorchNoBatchModelHandler(PytorchModelHandlerTensor):
    """
    At time of writing, we can't batch when using Beam's PytorchModelHandlerTensor.
    See https://github.com/apache/beam/blob/b221d804998734dc9025dadc0d8354562ca79c18/sdks/python/apache_beam/examples/inference/pytorch_language_modeling.py#L161
    """

    def batch_elements_kwargs(self):
        return {'max_batch_size': 1}


def filter_empty_lines(text: str) -> Iterator[str]:
    if len(text.strip()) > 0:
        yield text


def parse_args(argv):
    """Parses args for the workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        dest="input_path",
        required=True,
        help="Path to input JSONL.",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        required=True,
        help="Path to output JSONL.",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        required=True,
        help="Path to the model.",
    )
    parser.add_argument(
        "--id_field",
        dest="id_field",
        required=False,
        default="id",
        help="Name of ID field.",
    )
    return parser.parse_known_args(args=argv)


def run():
    """Run the classification pipeline using the RunInference API."""

    known_args, pipeline_args = parse_args(sys.argv)
    pipeline_options = PipelineOptions(pipeline_args)

    config = AutoConfig.from_pretrained(known_args.model_path)
    gen_fn = make_tensor_model_fn('forward')
    model_handler = PytorchNoBatchModelHandler(
        state_dict_path=os.path.join(known_args.model_path, "model.pth"),
        model_class=BertForSequenceClassification,
        model_params={
            "config": config,
        },
        device="cpu",
        inference_fn=gen_fn)
    keyed_model_handler = KeyedModelHandler(model_handler)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/allenai-specter")

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "ReadInputs" >> ReadFromText(known_args.input_path)
            | "FilterEmptyLines" >> beam.ParDo(filter_empty_lines)
            | "Parse JSON" >> beam.Map(json.loads)
            | "Tokenize" >> beam.ParDo(Tokenize(tokenizer=tokenizer, id_field=known_args.id_field))
            | "RunInference" >> RunInference(model_handler=keyed_model_handler)
            | "PostProcess" >> beam.ParDo(Postprocess(id_field=known_args.id_field))
            | "FormatOutput" >> beam.Map(json.dumps)
            | "WriteOutput" >> beam.io.WriteToText(known_args.output_path, ".jsonl")
        )


if __name__ == "__main__":
    run()
