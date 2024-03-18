# Setting up your environment.

## Basic Setup

You can use any semi-recent version of `torch` (v2) and huggingface `transformers`. Your default env will probably work just fine, except maybe needing `pip install pandas tqdm wandb`. If you want to test clip, you will also need `open_clip`. For the caption embedding you also need `sentence_transformers`

To experiment with discriminative ImageNet models, you will need `modelvshuman`. Please follow the instructions at https://github.com/bethgelab/model-vs-human.

For fancy colored logs, `pip install coloredlogs`.

## Details

For LLaVA use an individual environment:
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install pyarrow  # to disable pandas warning
```

For MoE-LLaVA use an individual conda enviroment:

```bash
git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
cd MoE-LLaVA
conda create -n moellava python=3.10 -y
conda activate moellava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
conda install cudatoolkit=11.8 -c pytorch
conda install -c conda-forge cudatoolkit-dev
export CUDA_HOME=$CONDA_PREFIX
pip install deepspeed
pip install flash-attn --no-build-isolation
pip install mpi4py
pip install pyarrow  # to disable pandas warning
```

## Hacks

COGvlm requires a different version of xformers
```bash
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

InterVL-Chat requires `fschat` and a few others but incorrectly asks the user to install packages that do not exist
```bash
pip install fschat
pip install flash_attn
```