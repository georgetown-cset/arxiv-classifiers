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
- name: cyber-full-weak-negs
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: Cyber with 10-pct Weak Negatives
      split: validation
      args: Cyber with 10-pct Weak Negatives
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9939791681779196
    - name: F1
      type: f1
      value: 0.7913030616772667
    - name: Precision
      type: precision
      value: 0.8110976349302608
    - name: Recall
      type: recall
      value: 0.7724516315333526
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cyber-full-weak-negs

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0203
- Accuracy: 0.9940
- F1: 0.7913
- Precision: 0.8111
- Recall: 0.7725

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
| 0.0171        | 1.0   | 37595  | 0.0190          | 0.9936   | 0.7866 | 0.7793    | 0.7941 |
| 0.013         | 2.0   | 75190  | 0.0178          | 0.9940   | 0.7915 | 0.8203    | 0.7647 |
| 0.0101        | 3.0   | 112785 | 0.0204          | 0.9940   | 0.7914 | 0.8109    | 0.7727 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
