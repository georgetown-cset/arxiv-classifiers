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
- name: ai-full-weak-negs
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: AI with 10-pct Weak Negatives
      split: validation
      args: AI with 10-pct Weak Negatives
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9664096469855303
    - name: F1
      type: f1
      value: 0.8936963215038082
    - name: Precision
      type: precision
      value: 0.8940826803566604
    - name: Recall
      type: recall
      value: 0.8933102964202797
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ai-full-weak-negs

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0952
- Accuracy: 0.9664
- F1: 0.8937
- Precision: 0.8941
- Recall: 0.8933

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
| 0.0803        | 1.0   | 37595  | 0.0873          | 0.9664   | 0.8939 | 0.8914    | 0.8964 |
| 0.07          | 2.0   | 75190  | 0.0862          | 0.9667   | 0.8946 | 0.8963    | 0.8929 |
| 0.0613        | 3.0   | 112785 | 0.0954          | 0.9664   | 0.8937 | 0.8940    | 0.8934 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
