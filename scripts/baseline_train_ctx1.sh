#!/bin/bash
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE RUNNABLE!
#SBATCH --job-name=baselines1
#SBATCH --err=baselines1_.stderr
#SBATCH --output=baselines1_.stdout
#SBATCH --mail-type=END
#SBATCH --mail-user=aostapen@andrew.cmu.edu
#SBATCH --mem 128gb
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --gres=gpu:1
#SBATCH --time=0
INITIAL_DIR=$(pwd)
BASE_DIR=""
MODEL_DIR=""


function log {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] - LOGGER - $@"
}


function notify_error {
    echo "ERROR EXECUTING COMMAND"
    kill_ssh_tunnel
    cd $INITIAL_DIR
    exit 1
}


trap notify_error ERR

source ~/.bashrc
log "Running on node $(hostname) ..."

log "Activating Conda environment ..."
# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate ${CONDA_ENV}
conda activate p36-clean
export TOKENIZERS_PARALLELISM=false

# ===== Call your code below =====
log "context size 1"
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 18 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 42 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 614 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2

log "context size 1"
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 212 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 123 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 720 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2

log "context size 1"
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 333 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 444 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 555 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2

log "context size 1"
python3 run_baseline.py --context_size 1 --dataset_basedir bangor_data/feature_spk_new_raw --gpus 1 --no_adapter --max_epochs 10 --batch_size 16 --balanced --lr 1e-5 --seed 981 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --tensorboard_dir tb_base2
