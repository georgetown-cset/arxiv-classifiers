---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- arxiv
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: cv-32
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: CV
      split: validation
      args: CV
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9866654149936207
    - name: F1
      type: f1
      value: 0.8729623155412821
    - name: Precision
      type: precision
      value: 0.8817442719881744
    - name: Recall
      type: recall
      value: 0.8643535662534213
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cv-32

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0460
- Accuracy: 0.9867
- F1: 0.8730
- Precision: 0.8817
- Recall: 0.8644

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7.5e-06
- train_batch_size: 32
- eval_batch_size: 32
- seed: 20230227
- distributed_type: tpu
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step   | Validation Loss | Accuracy | F1     | Precision | Recall |
|:-------------:|:-----:|:------:|:---------------:|:--------:|:------:|:---------:|:------:|
| 0.0433        | 1.0   | 34177  | 0.0358          | 0.9868   | 0.8752 | 0.8771    | 0.8733 |
| 0.0294        | 2.0   | 68354  | 0.0386          | 0.9868   | 0.8763 | 0.8677    | 0.8851 |
| 0.0228        | 3.0   | 102531 | 0.0461          | 0.9867   | 0.8732 | 0.8821    | 0.8645 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
