# Background

This repo contains code and data for training classifiers on [arXiv category labels](https://arxiv.org/category_taxonomy).
We predict the label(s) assigned to a paper using its title and abstract text.
An earlier effort to do this is described [here](https://arxiv.org/abs/2002.07143).
We deploy via Airflow using the DAG defined in [arxiv_classifier_dag.py](arxiv_classifier_dag.py).

# Overview

## Data

* We extracted arXiv paper metadata from BQ (`gcp_cset_arxiv_metadata.arxiv_metadata_latest`) to GCS (`gs://arxiv-classifier-v2/source/`). See [`extract.sh`](dataset/scripts/extract.sh).
* We downloaded it to disk under `data/source`. See [`download.sh`](dataset/scripts/download.sh).
* We preprocessed and partitioned the data into 70/15/15 train/eval/test splits. See [`split.py`](dataset/scripts/split.py).
* We make this data available for training as a 🤗 Dataset from `data/dataset/arxiv`. See [`arxiv.py`](dataset/arxiv/arxiv.py).

## Training

* We used [PyTorch/XLA](https://github.com/pytorch/xla) to train on TPU.

> PyTorch/XLA is a Python package that uses the XLA deep learning compiler to connect the PyTorch deep learning framework and Cloud TPUs.

* Our first abstraction layer over PyTorch/XLA is [Huggingface Transformers](https://huggingface.co/docs/transformers/main/en/index).
* And to adapt our training script for use on TPU, we used [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index).
* [Weights & Biases](https://wandb.ai/cset) provided monitoring.

# Setup

The Cloud TPU docs have a [user guide](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm) for PyTorch/XLA.
The first thing it says is that you can use either TPU VMs or TPU Pods, but they (now) recommend VMs, so we used one.

### Create a TPU VM

We created a TPU VM `arxiv-classifier-v2` in `us-central1-a` ([console](https://console.cloud.google.com/compute/tpus/details/us-central1-a/arxiv-classifier-v2?project=gcp-cset-projects&tab=details)):

```shell
gcloud compute tpus tpu-vm create arxiv-classifier-v2 \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-1.13
```

This VM is backed by a [v3.8 TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3) and nominally costs [$8/hour](https://cloud.google.com/products/calculator/#id=8257832f-5226-4273-9817-6799f37b9e83).
At time of writing, v4 TPUs were limited-availability. PyTorch 1.13 is pre-installed with CUDA 11.7.

### Configure the TPU VM

⚠️ We stop and start the TPU VM like
```shell
gcloud compute tpus tpu-vm start arxiv-classifier-v2 --zone=us-central1-a
gcloud compute tpus tpu-vm stop arxiv-classifier-v2 --zone=us-central1-a
```

Shell in like
```shell
gcloud compute tpus tpu-vm ssh arxiv-classifier-v2 --zone=us-central1-a
```

⚠️ At time of writing, the `gcloud compute tpus` CLI requires application default credentials (ADC) set.
Otherwise, with user auth alone, you may see this:
```shell
% gcloud compute tpus tpu-vm start arxiv-classifier-v2 --zone=us-central1-a
Request issued for: [arxiv-classifier-v2]
Waiting for operation [projects/gcp-cset-projects/locations/us-central1-a/operations/operation-1669841256413-5eeb636eb799d-6070f791-4abbb44a] to complete...failed.
ERROR: (gcloud.compute.tpus.tpu-vm.start) {
  "code": 8,
  "message": "There is no more capacity in the zone \"us-central1-a\"; you can try in another zone where Cloud TPU Nodes are offered (see https://cloud.google.com/tpu/docs/regions) [EID: 0x191e4b746b2c69a9]"
}
```

The above is solved by running `gcloud auth application-default login`.

⚠️ We ran into issues when logging into the VM with more than one Unix user account.
Accelerate creates directories (e.g. for logging) outside `~/` as the first user which aren't writeable by a second user.

### Verify PyTorch/XLA has TPU access

Per the PyTorch/XLA user guide, we don't bother with virtualenvs on the TPU VM.
This minimal example from the user guide just works:
```
% export XRT_TPU_CONFIG="localservice;0;localhost:51011"
% python3
Python 3.8.10 (default, Mar 15 2022, 12:22:08)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import torch_xla.core.xla_model as xm
>>> dev = xm.xla_device()
>>> t1 = torch.randn(3,3,device=dev)
>>> t2 = torch.randn(3,3,device=dev)
>>> print(t1 + t2)
tensor([[-1.1846, -0.7140, -0.4168],
        [-0.3259, -0.5264, -0.8828],
        [-0.8562, -0.5813,  0.3264]], device='xla:1')
```

⚠️ Before each training session, it's necessary to
```shell
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```
You may want to add that to your `.profile`.

### Loading resources on the TPU VM

We can `scp` resources to the TPU VM via `gcloud compute tpus tpu-vm scp`, e.g. like this
```shell
gcloud compute tpus tpu-vm scp --recurse ./training arxiv-classifier-v2:~/ --zone=us-central1-a
```
To copy a local directory `training` and its contents to the user directory on our TPU VM.

### Storage

We have a project-specific bucket `gs://arxiv-classifier-v2`.

[Per the TPU docs](https://cloud.google.com/tpu/docs/storage-buckets), we created a service agent for the TPU VM to access the project bucket:
```shell
$ gcloud beta services identity create --service tpu.googleapis.com
Service identity created: service-855475113448@cloud-tpu.iam.gserviceaccount.com
```

Service _agents_ are slightly different from service _accounts_ and not as well-documented.
Anyway, the key thing is we granted `service-855475113448@cloud-tpu.iam.gserviceaccount.com` the Storage Admin role for our project bucket using the [console](https://console.cloud.google.com/storage/browser/arxiv-classifier-v2;tab=permissions?forceOnBucketsSortingFiltering=false&project=gcp-cset-projects&prefix=&forceOnObjectsSortingFiltering=false).

From the VM, we can confirm there's access to the project bucket.
```shell
touch test.tmp
gsutil cp ./test.tmp gs://arxiv-classifier-v2/
gsutil rm gs://arxiv-classifier-v2/test.tmp
```

### Training script

Our training script is `training/run.py`.
We invoke it using the Huggingface [accelerate](https://huggingface.co/docs/accelerate/index) library.
Configurations for training run are in JSON files under `training/configs`.
Invoking the script from `training/` with a config at `debug.json` looks like
```shell
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TOKENIZERS_PARALLELISM=False
accelerate launch run.py configs/debug.json
```

The config JSON contains all of our arguments for the training script, which we adapted from the [GLUE example script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

### 🤗 Accelerate

Before the first training run, we set up Accelerate as below [using](https://huggingface.co/docs/accelerate/package_reference/cli) `accelerate config`:

```shell
$ accelerate config
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 3
What is the name of the function in your script that should be launched in all parallel scripts? [main]:
Are you using a TPU cluster? [yes/NO]:
How many TPU cores should be used for distributed training? [1]:
```

This creates a config file at `~/.cache/huggingface/accelerate/default_config.yaml`.

We then edit the values of `downcast_bf16` and `mixed_precision` to `yes` as below:
```yaml
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: TPU
downcast_bf16: 'yes'
fsdp_config: {}
gpu_ids: null
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'yes'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```

Accelerate supports [some](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp) [interesting](https://huggingface.co/docs/accelerate/main/en/usage_guides/megatron_lm) training strategies.
Its "[TPU Best Practices](https://huggingface.co/docs/accelerate/main/en/concept_guides/training_tpu)" guide is very helpful.

For example,

> As you launch your script, you may notice that training seems exceptionally slow at first. This is because TPUs first run through a few batches of data to see how much memory to allocate before finally utilizing this configured memory allocation extremely efficiently.
>
> If you notice that your evaluation code to calculate the metrics of your model takes longer due to a larger batch size being used, it is recommended to keep the batch size the same as the training data if it is too slow. Otherwise the memory will reallocate to this new batch size after the first few iterations.

## Workflow

### First-time environment setup

Start the TPU VM if it isn't already running:
```shell
gcloud compute tpus tpu-vm start arxiv-classifier-v2 --zone=us-central1-a
```

Shell in:
```shell
gcloud compute tpus tpu-vm ssh arxiv-classifier-v2 --zone=us-central1-a
```

Clone this repo:
```shell
git clone https://github.com/georgetown-cset/arxiv-classifier-v2.git
```

Download the source data:
```shell
cd arxiv-classifier-v2
cd dataset/scripts
bash download.sh
```

Create pre-processed splits:
```shell
PYTHONPATH=../.. python3 split.py
```

The dataset is ready for training.

### Define and run an experiment

We define experiments using JSON config files in the `training/configs` directory.

This file specifies parameters including:
- `model_name_or_path`: The name of a pretrained model from the [Huggingface hub](https://huggingface.co/models), or a local path for a pretrained model. We're using `sentence-transformers/allenai-specter`.
- `dataset_name`: The name of a dataset defined in the [`arXiv.py`](dataset/arxiv/arxiv.py) dataset load script, specifically in any of the `BuilderConfig` classes listed in the `BUILDER_CONFIGS` attribute of the `ArxivDataset` classes. For example: `AI` or or `AI Undersample 4-to-1`.
- `run_name`: The name of the training run for monitoring in Weights & Biases.
- `output_dir`: Where to save training outputs (checkpoints, stats, configs, and the trained model). We're using subdirectories of `./training/models`.
- `max_train_samples`: For debugging purposes, a limit on the number of examples to use from the `train` split.
- `max_eval_samples`: Similarly, a limit on the examples to use from the `validation` split.
- `evaluation_strategy`: Either `steps` or `epoch`, indicating when to evaluate the model during training.
- `eval_steps`: If `steps` above, how often to evaluate (e.g. `500` to evaluate after every 500 steps).

After creating a new config file:
- Commit it and push to GitHub
- `git pull` from the TPU VM

On the VM, we start an experiment with `accelerate launch {script-path} {config-path}`.

Using the [`debug.json`](training/configs/debug.json) config looks like this:
```shell
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TOKENIZERS_PARALLELISM=False
accelerate launch training/run.py training/configs/debug.json
```

One way to explore the available model, training, and accelerate parameters is via
```shell
accelerate launch training/run.py --help
```

## Models

The [`./models`](models) directory contains a subset of the models trained selected from [wandb](https://wandb.ai/cset/huggingface?workspaceuser-jamesdunham).
Each of these models was trained on all of arXiv.

- ai-32: AI classifier.
- ai-full-weak-negs: AI classifier trained on arXiv and OpenAlex weak-negative examples 10% the size of arXiv.
- cv-32: CV subject classifier.
- cyber-32: Cyber subject classifier.
- cyber-full-weak-negs: Cyber subject classifier trained on arXiv and OpenAlex weak-negative examples 10% the size of arXiv.
- nlp-32: NLP subject classifier.
- ro-32: Robotics subject classifier.

## Deployment

We deploy on Dataflow using custom containers that provide large model assets and pre-installed dependencies.
The container runs on each worker during inference, so each worker has what it needs without long setup times or downloads.
Docs on this are [here](https://cloud.google.com/dataflow/docs/guides/using-custom-containers).

### arxiv-classifier-v2 containers

The container is defined by our [Dockerfile](./Dockerfile).
It sets up a Python environment correctly, then copies in our model assets from the [`./models`](models) directory.

Beam's RunInference API (see below) requires `.pth` or state-dict-format PyTorch binaries, so we've run `./inference/to_state_dict.py` to produce these from selected model checkpoint files, writing `.pth` binaries under the [./images/](images) directory, where they're copied into the container images on build.

⚠️ Snippets below assume we've run:
```shell
export REPO=arxiv-classifier-v2
export IMAGE=ai
export IMAGE_URI=us-east1-docker.pkg.dev/gcp-cset-projects/$REPO/$IMAGE:latest
```

To update and test a container, you can build it locally:
```shell
docker build . --tag $IMAGE_URI
docker run --name ai-classifier-v2-local --entrypoint /bin/bash -it $IMAGE_URI
```

To make containers available for Dataflow jobs, we use Cloud Build, which [builds](https://console.cloud.google.com/cloud-build/builds?project=gcp-cset-projects)
each image on GCP and pushes it to an Artifact Registry
[repo](https://console.cloud.google.com/artifacts/docker/gcp-cset-projects/us-east1/arxiv-classifier-v2?project=gcp-cset-projects)
called `arxiv-classifier-v2`.

```shell
gcloud builds submit . --tag $IMAGE_URI
```
At time of writing, building the container takes ~30 minutes after uploading the assets.
We minimize the container size by adding files that aren't needed in deployment to [`.gcloudignore`](./.gcloudignore).

You can also deploy the custom container to a GCE instance for testing like so:
```shell
gcloud compute instances create-with-container arxiv-classifier-v2-deployment-testing \
  --project=gcp-cset-projects \
  --zone=us-east1-c \
  --machine-type=n2-standard-2 \
  --network-interface=network-tier=PREMIUM,subnet=default \
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD \
  --service-account=855475113448-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
  --image=projects/cos-cloud/global/images/cos-stable-101-17162-127-42 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --boot-disk-device-name=arxiv-classifier-v2-deployment-testing \
  --container-image=us-east1-docker.pkg.dev/gcp-cset-projects/arxiv-classifier-v2/ai:latest \
  --container-restart-policy=always \
  --container-privileged \
  --container-stdin \
  --container-tty \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --labels=ec-src=vm_add-gcloud,container-vm=cos-stable-101-17162-127-42
```

### Dataflow

Scripts and resources for inference are in the [`./inference`](inference) directory.
Deploying on Dataflow uses the [`./inference/beam_runner.py`](beam_runner.py) pipeline script and Beam's [RunInference API](https://beam.apache.org/documentation/ml/about-ml/#use-runinference).

We extract the merged corpus to GCS with [extract-corpus.sh](inference/extract-corpus.sh).
Via `bq extract` this produces `gs://arxiv-classifier-v2/dataflow/merged-corpus-input/merged-corpus-input-\*.jsonl`.
These files are given as the `--input_path` on Dataflow.

There's a series of bash scripts for submitting jobs to Dataflow:
- [`./inference/run.py`](run-merged-corpus.py): Inference on the merged corpus using the AI model.
- [`./inference/run.py`](run-merged-corpus-cyber.py): Inference on the merged corpus using the cyber subject model.
- [`./inference/run.py`](run-merged-corpus-nlp.py): Inference on the merged corpus using the NLP subject model.

Outputs are written to `gs://arxiv-classifier-v2/dataflow/merged-corpus-output/$MODEL/merged-corpus-predictions-`.

For small-scale testing, see [`./inference/test-dataflow.sh`](test-dataflow.sh).

Cost estimation is available through [`./inference/estimate_cost.py`](estimate_cost.py).

### Locally

There's also a predictions script for smallish local input without Beam, [`inference/run.py`](run.py).

And for testing on DirectRunner, [`./inference/test-directrunner.sh`](test-directrunner.sh).
