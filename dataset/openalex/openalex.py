"""
This is a local loading script for GPT-labeled classification datasets.

Use like:
    dataset = load_dataset("dataset/openalex/openalex.py", split="train")

Where `split` identifies one of the dataset variants defined by a `ArXivConfig` below.

The first 🤗 resource linked below, the conceptual guide "Build and load", is particularly helpful for getting up to
speed on what's happening here.

References:
    - https://huggingface.co/docs/datasets/about_dataset_load
    - https://huggingface.co/docs/datasets/loading#local-loading-script
"""
from typing import Iterable, List, Union

import datasets
import pandas as pd
import pyarrow.dataset as ds
from datasets.tasks import TextClassification
from imblearn.under_sampling import RandomUnderSampler

_CITATION = "TODO"

_DESCRIPTION = """
 arXiv classification dataset. Contains paper titles, abstracts, and category labels extracted from ``gcp_cset_arxiv_metadata.arxiv_metadata_latest``.
 We've restricted the data to papers since 2010 and applied some preprocessing.
"""

_SEED = 20230216

# These are the arXiv categories we consider AI-relevant
AI_LABELS = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.MA", "cs.RO", "stat.ML"]

# Map arXiv categories to multi-label labels for multi-label solutions
AI_MULTILABEL = {
    "cs.AI": "AI",
    "cs.CL": "NLP",
    "cs.CR": "Cyber",
    "cs.CV": "CV",
    "cs.LG": "AI",
    "cs.MA": "AI",
    "cs.RO": "RO",
    "stat.ML": "AI",
}


class ArXivConfig(datasets.BuilderConfig):

    def __init__(self,
                 name,
                 labels,
                 undersample_ratio=None,
                 sample_pct=None,
                 weak_neg_pct=None,
                 multilabel=False,
                 **kwargs):
        """
        BuilderConfig for arXiv classification datasets.

        This is a helper class that allows quick specification of dataset variants. After it's used to create an
        ArxivDataset instance, the ArXivConfig instance is available as the `config` attribute.

        Parameters
        ----------
        name : str
            Split name.
        labels : list | dict
            Label specification using arXiv categories. (See `AI_LABELS` and `AI_MULTILABEL`.)
        undersample_ratio : Optional[float]
            Passed to `imblearn.under_sampling.RandomUnderSampler` for undersampling of the majority class in the train
            split. For example, `4` will undersample to achieve a 1:4 ratio of minority- to majority-class examples.
        sample_pct : Optional[int]
            When in (0, 100), the data is undersampled to the specified percentage size to achieve smaller splits.
        weak_neg_pct : Optional[float]
            When a number in (0, 1), the training split is augmented with weakly negative non-arXiv examples.
        multilabel : bool
            When True, expect labels in the data for multilabel classification.
        kwargs : dict
            Additional keyword arguments  passed to BuilderConfig.
        """
        super(ArXivConfig, self).__init__(
            name,
            description="arXiv Classification Dataset",
            version=datasets.Version("0.0.1"),
            **kwargs)
        if isinstance(labels, list):
            # We're doing binary classification and have a list of arXiv categories that identify positive examples
            self.labels = {category: "True" for category in labels}
        elif isinstance(labels, dict):
            # We're taking a multi-label approach and instead have a mapping of arXiv categories to labels
            self.labels = labels
        else:
            raise ValueError(labels)
        self.undersample_ratio = undersample_ratio
        self.sample_pct = sample_pct
        self.weak_neg_pct = weak_neg_pct
        self.multilabel = multilabel


class ArxivDataset(datasets.GeneratorBasedBuilder):
    """ArxivClassification Dataset."""

    # We load the dataset from a local directory containing these files
    _DOWNLOAD_URL = "."
    _TRAIN_FILE = "train.parquet"
    _VAL_FILE = "valid.parquet"
    _TEST_FILE = "test.parquet"
    _NEGATIVES_FILE = "weak_negatives.parquet"

    # ArXivConfig builder configs define how the data is loaded and labeled
    BUILDER_CONFIGS = [
        # AI and subject models
        # The `labels` param specifies which arXiv categories represent our positive labels
        ArXivConfig("AI", AI_LABELS),
        ArXivConfig("CV", ["cs.CV"]),
        ArXivConfig("NLP", ["cs.CL"]),
        ArXivConfig("RO", ["cs.RO"]),
        ArXivConfig("Cyber", ["cs.CR"]),

        # 10% subsample
        ArXivConfig("AI 10-pct Subsample", AI_LABELS, sample_pct=10),
        ArXivConfig("CV 10-pct Subsample", ["cs.CV"], sample_pct=10),
        ArXivConfig("NLP 10-pct Subsample", ["cs.CL"], sample_pct=10),
        ArXivConfig("RO 10-pct Subsample", ["cs.RO"], sample_pct=10),
        ArXivConfig("Cyber 10-pct Subsample", ["cs.CR"], sample_pct=10),

        # 1% subsample
        ArXivConfig("AI 1-pct Subsample", AI_LABELS, sample_pct=1),
        ArXivConfig("CV 1-pct Subsample", ["cs.CV"], sample_pct=1),
        ArXivConfig("NLP 1-pct Subsample", ["cs.CL"], sample_pct=1),
        ArXivConfig("RO 1-pct Subsample", ["cs.RO"], sample_pct=1),
        ArXivConfig("Cyber 1-pct Subsample", ["cs.CR"], sample_pct=1),

        # Random undersampling of the majority class to achieve 4:1 class imbalance
        ArXivConfig("AI Undersample 4-to-1", AI_LABELS, undersample_ratio=4),
        ArXivConfig("CV Undersample 4-to-1", ["cs.CV"], undersample_ratio=4),
        ArXivConfig("NLP Undersample 4-to-1", ["cs.CL"], undersample_ratio=4),
        ArXivConfig("RO Undersample 4-to-1", ["cs.RO"], undersample_ratio=4),
        ArXivConfig("Cyber Undersample 4-to-1", ["cs.CR"], undersample_ratio=4),

        # Random undersampling of the majority class to achieve 2:1 class imbalance
        ArXivConfig("AI Undersample 2-to-1", AI_LABELS, undersample_ratio=2),
        ArXivConfig("CV Undersample 2-to-1", ["cs.CV"], undersample_ratio=2),
        ArXivConfig("NLP Undersample 2-to-1", ["cs.CL"], undersample_ratio=2),
        ArXivConfig("RO Undersample 2-to-1", ["cs.RO"], undersample_ratio=2),
        ArXivConfig("Cyber Undersample 2-to-1", ["cs.CR"], undersample_ratio=2),

        # Random undersampling of the majority class to achieve 4:1 class imbalance following a 10% subsample
        # This starts to look like copypasta, but it's our reference for which dataset variants are available for use,
        # and defining & naming the variants programmatically is a lot less readable.
        ArXivConfig("AI Undersample 4-to-1 with 10-pct Subsample", AI_LABELS, undersample_ratio=4, sample_pct=10),
        ArXivConfig("CV Undersample 4-to-1 with 10-pct Subsample", ["cs.CV"], undersample_ratio=4, sample_pct=10),
        ArXivConfig("NLP Undersample 4-to-1 with 10-pct Subsample", ["cs.CL"], undersample_ratio=4, sample_pct=10),
        ArXivConfig("RO Undersample 4-to-1 with 10-pct Subsample", ["cs.RO"], undersample_ratio=4, sample_pct=10),
        ArXivConfig("Cyber Undersample 4-to-1 with 10-pct Subsample", ["cs.CR"], undersample_ratio=4, sample_pct=10),

        # Random undersampling of the majority class to achieve 2:1 class imbalance following a 10% subsample
        ArXivConfig("AI Undersample 2-to-1 with 10-pct Subsample", AI_LABELS, undersample_ratio=2, sample_pct=10),
        ArXivConfig("CV Undersample 2-to-1 with 10-pct Subsample", ["cs.CV"], undersample_ratio=2, sample_pct=10),
        ArXivConfig("NLP Undersample 2-to-1 with 10-pct Subsample", ["cs.CL"], undersample_ratio=2, sample_pct=10),
        ArXivConfig("RO Undersample 2-to-1 with 10-pct Subsample", ["cs.RO"], undersample_ratio=2, sample_pct=10),
        ArXivConfig("Cyber Undersample 2-to-1 with 10-pct Subsample", ["cs.CR"], undersample_ratio=2, sample_pct=10),

        # Random undersampling of the majority class to achieve 4:1 class imbalance following a 1% subsample
        ArXivConfig("AI Undersample 4-to-1 with 1-pct Subsample", AI_LABELS, undersample_ratio=4, sample_pct=1),
        ArXivConfig("CV Undersample 4-to-1 with 1-pct Subsample", ["cs.CV"], undersample_ratio=4, sample_pct=1),
        ArXivConfig("NLP Undersample 4-to-1 with 1-pct Subsample", ["cs.CL"], undersample_ratio=4, sample_pct=1),
        ArXivConfig("RO Undersample 4-to-1 with 1-pct Subsample", ["cs.RO"], undersample_ratio=4, sample_pct=1),
        ArXivConfig("Cyber Undersample 4-to-1 with 1-pct Subsample", ["cs.CR"], undersample_ratio=4, sample_pct=1),

        # Random undersampling of the majority class to achieve 2:1 class imbalance following a 1% subsample
        ArXivConfig("AI Undersample 2-to-1 with 1-pct Subsample", AI_LABELS, undersample_ratio=2, sample_pct=1),
        ArXivConfig("CV Undersample 2-to-1 with 1-pct Subsample", ["cs.CV"], undersample_ratio=2, sample_pct=1),
        ArXivConfig("NLP Undersample 2-to-1 with 1-pct Subsample", ["cs.CL"], undersample_ratio=2, sample_pct=1),
        ArXivConfig("RO Undersample 2-to-1 with 1-pct Subsample", ["cs.RO"], undersample_ratio=2, sample_pct=1),
        ArXivConfig("Cyber Undersample 2-to-1 with 1-pct Subsample", ["cs.CR"], undersample_ratio=2, sample_pct=1),

        # 10% weak negatives
        ArXivConfig("AI with 10-pct Weak Negatives", AI_LABELS, weak_neg_pct=10),
        ArXivConfig("CV with 10-pct Weak Negatives", ["cs.CV"], weak_neg_pct=10),
        ArXivConfig("NLP with 10-pct Weak Negatives", ["cs.CL"], weak_neg_pct=10),
        ArXivConfig("RO with 10-pct Weak Negatives", ["cs.RO"], weak_neg_pct=10),
        ArXivConfig("Cyber with 10-pct Weak Negatives", ["cs.CR"], weak_neg_pct=10),

        # 10% subsample with 10% weak negatives
        ArXivConfig("AI 10-pct Subsample with 10-pct Weak Negatives", AI_LABELS, sample_pct=10, weak_neg_pct=10),
        ArXivConfig("CV 10-pct Subsample with 10-pct Weak Negatives", ["cs.CV"], sample_pct=10, weak_neg_pct=10),
        ArXivConfig("NLP 10-pct Subsample with 10-pct Weak Negatives", ["cs.CL"], sample_pct=10, weak_neg_pct=10),
        ArXivConfig("RO 10-pct Subsample with 10-pct Weak Negatives", ["cs.RO"], sample_pct=10, weak_neg_pct=10),
        ArXivConfig("Cyber 10-pct Subsample with 10-pct Weak Negatives", ["cs.CR"], sample_pct=10, weak_neg_pct=10),

        # Multilabel format, not successfully implemented
        ArXivConfig("Multilabel with 1-pct Subsample", AI_MULTILABEL, sample_pct=1, multilabel=True)
    ]

    def _info(self):
        # Define DatasetInfo
        features = {
            "text": datasets.Value("string"),
            "id": datasets.Value("string"),
        }
        if self.config.multilabel:
            raise NotImplementedError
            features["label"] = datasets.features.Sequence(datasets.features.ClassLabel(names=self._get_label_names()))
        else:
            features["label"] = datasets.features.ClassLabel(names=self._get_label_names())

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="label"),
            ],
        )

    def _get_label_names(self):
        # Retrieve the label names for DatasetInfo reporting (they vary with the builder config that was used)
        positive_labels = set(self.config.labels.values())
        if self.config.multilabel:
            return list(sorted(list(positive_labels)))
        else:
            # Putting "False" first makes its ClassLabel integer 0
            return ["False"] + sorted(list(positive_labels))

    def _split_generators(self, dl_manager):
        # Define train/valid/test splits using local filepaths; when called, invokes caching
        train_path = dl_manager.download_and_extract(self._TRAIN_FILE)
        val_path = dl_manager.download_and_extract(self._VAL_FILE)
        test_path = dl_manager.download_and_extract(self._TEST_FILE)
        negatives_path = None
        if self.config.weak_neg_pct:
            negatives_path = dl_manager.download_and_extract(self._NEGATIVES_FILE)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split_key": "train", "filepath": train_path, "negatives_path": negatives_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"split_key": "validation", "filepath": val_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"split_key": "test", "filepath": test_path}
            ),
        ]

    def _category_to_label(self, categories: Iterable[str]) -> Union[str, List[str]]:
        #
        multilabels = []
        for category in categories:
            label = self.config.labels.get(category, "")
            if label and not self.config.multilabel:
                return label
            elif label and self.config.multilabel:
                multilabels.append(label)
        if self.config.multilabel:
            return multilabels
        return "False"

    def _generate_examples(self, split_key, filepath, negatives_path=None):
        # We iterate over chunks of the PyArrow dataset cached on the disk after _split_generators is called
        # https://huggingface.co/docs/datasets/about_dataset_load#datasets-datasetbuilder
        # https://huggingface.co/docs/datasets/about_arrow
        chunks = ds.dataset(filepath, format="parquet")
        # If `weak_neg_pct` was passed, we'll iterate over another dataset of weak-negative examples at the same time
        if split_key == "train" and self.config.weak_neg_pct:
            negatives = pd.read_parquet(negatives_path).sample(frac=1, random_state=_SEED)
            if self.config.multilabel:
                negatives["label"] = []
            else:
                negatives["label"] = "False"
            neg_offset = 0
        # Ref: https://arrow.apache.org/docs/python/dataset.html#iterative-out-of-core-or-streaming-reads
        negs_exhausted = False
        for i, chunk in enumerate(chunks.to_batches(columns=["id", "categories", "text"])):
            # To create labels, undersample, etc. on the fly, we read each chunk into a pandas dataset
            records: pd.DataFrame = chunk.to_pandas()
            if self.config.sample_pct:
                # Downsample the chunk to the specified size
                records = records.sample(frac=self.config.sample_pct / 100, random_state=_SEED)
            # Create label column
            records["label"] = records["categories"].apply(self._category_to_label)
            if split_key == "train" and self.config.undersample_ratio:
                # We're addressing the class imbalance with naive undersampling in the train split
                sampler = RandomUnderSampler(random_state=_SEED, sampling_strategy=1 / self.config.undersample_ratio)
                records, _ = sampler.fit_resample(records, records["label"])
            if split_key == "train" and self.config.weak_neg_pct and not negs_exhausted:
                # We're augmenting the data with weak negative examples in the train split. The builder config includes
                # a `weak_neg_pct` int like `10`, indicating we want to draw a sample of weak negative examples 10% the
                # size of the original data. Below, since we're doing this chunk-by-chunk, we calculate how many records
                # is e.g. 10% of the current chunk, and then ensure that the result is at least zero records and not
                # more than the number of weak-negative examples available.
                neg_sample_size = max(0,
                                      min(
                                          round((self.config.weak_neg_pct / 100) * records.shape[0]),
                                          negatives.shape[0],
                                      ))
                if neg_offset + neg_sample_size > negatives.shape[0]:
                    # We iterate over slices of the weak negatives dataset chunk-by-chunk by keeping track of an offset.
                    # If we reach the end of our weak negatives dataset doing this, we stop adding them for any
                    # remaining chunks. In practice this happens when training on the full data using weak negatives.
                    print("Weak negatives exhausted")
                    negs_exhausted = True
                neg_sample = negatives.iloc[(neg_offset):(neg_offset + neg_sample_size)]
                print(f"Adding {neg_sample.shape[0]} weak negatives from {neg_offset}:{(neg_offset + neg_sample_size)} "
                      f"to chunk with size {records.shape[0]}")
                neg_offset += neg_sample_size
                # Append the weak negatives to the arXiv data and shuffle the result
                records = pd.concat([records, neg_sample]).sample(frac=1, random_state=_SEED).drop_duplicates("id")
            for id_, label, text in zip(records["id"], records["label"], records["text"]):
                yield id_, {"text": text, "label": label, "id": id_}
