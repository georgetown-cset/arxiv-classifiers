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
- name: ai-32
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: AI
      split: validation
      args: AI
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9667638135632998
    - name: F1
      type: f1
      value: 0.8948214165147524
    - name: Precision
      type: precision
      value: 0.8951719666063275
    - name: Recall
      type: recall
      value: 0.8944711408671238
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ai-32

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0938
- Accuracy: 0.9668
- F1: 0.8948
- Precision: 0.8952
- Recall: 0.8945

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
| 0.0903        | 1.0   | 34177  | 0.0844          | 0.9659   | 0.8887 | 0.9193    | 0.8601 |
| 0.075         | 2.0   | 68354  | 0.0867          | 0.9670   | 0.8953 | 0.8992    | 0.8914 |
| 0.0604        | 3.0   | 102531 | 0.0940          | 0.9668   | 0.8949 | 0.8953    | 0.8946 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
