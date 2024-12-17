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
- name: cyber-32
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: arxiv
      type: arxiv
      config: Cyber
      split: validation
      args: Cyber
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9938724914978686
    - name: F1
      type: f1
      value: 0.7867537867537867
    - name: Precision
      type: precision
      value: 0.8098440843778661
    - name: Recall
      type: recall
      value: 0.7649436904418134
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cyber-32

This model is a fine-tuned version of [sentence-transformers/allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) on the arxiv dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0215
- Accuracy: 0.9939
- F1: 0.7868
- Precision: 0.8098
- Recall: 0.7649

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
| 0.0192        | 1.0   | 34177  | 0.0168          | 0.9938   | 0.7764 | 0.8312    | 0.7283 |
| 0.0125        | 2.0   | 68354  | 0.0186          | 0.9939   | 0.7804 | 0.8341    | 0.7332 |
| 0.0104        | 3.0   | 102531 | 0.0215          | 0.9939   | 0.7865 | 0.8095    | 0.7647 |


### Framework versions

- Transformers 4.27.0.dev0
- Pytorch 1.13.0+cu117
- Datasets 2.7.1
- Tokenizers 0.13.2
