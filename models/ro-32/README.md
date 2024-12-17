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
- name: ro-32
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: RO
      split: validation
      args: RO
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9950800715160463
    - name: F1
      type: f1
      value: 0.8026698613725826
    - name: Precision
      type: precision
      value: 0.8389982110912343
    - name: Recall
      type: recall
      value: 0.7693569553805775
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ro-32

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0189
- Accuracy: 0.9951
- F1: 0.8027
- Precision: 0.8390
- Recall: 0.7694

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
| 0.0192        | 1.0   | 34177  | 0.0148          | 0.9949   | 0.8038 | 0.8071    | 0.8005 |
| 0.0133        | 2.0   | 68354  | 0.0167          | 0.9951   | 0.8078 | 0.8211    | 0.7949 |
| 0.0083        | 3.0   | 102531 | 0.0190          | 0.9951   | 0.8032 | 0.8402    | 0.7694 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
