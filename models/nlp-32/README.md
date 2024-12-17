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
- name: nlp-32
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: NLP
      split: validation
      args: NLP
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9934201823744522
    - name: F1
      type: f1
      value: 0.8574334319526628
    - name: Precision
      type: precision
      value: 0.8839115516584064
    - name: Recall
      type: recall
      value: 0.8324955116696588
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# nlp-32

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0252
- Accuracy: 0.9934
- F1: 0.8574
- Precision: 0.8839
- Recall: 0.8325

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
| 0.018         | 1.0   | 34177  | 0.0187          | 0.9933   | 0.8542 | 0.8784    | 0.8312 |
| 0.0138        | 2.0   | 68354  | 0.0215          | 0.9935   | 0.8581 | 0.8938    | 0.8251 |
| 0.0122        | 3.0   | 102531 | 0.0253          | 0.9934   | 0.8572 | 0.8839    | 0.8321 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2