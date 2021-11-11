#!/bin/bash
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE RUNNABLE!
#SBATCH --job-name=partner-models1
#SBATCH --err=partner_models1.stderr
#SBATCH --output=partner_models1.stdout
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
# ===== Call your code below =====
log "context size 1 -- seeds 18, 42, 614"
python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 18 --tensorboard_dir partner_tb/ --order --leave_one_out
#python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 42 --tensorboard_dir partner_tb/
#python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 614 --tensorboard_dir partner_tb/

log "context size 1 -- seeds 212, 123, 720"
#python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 212 --tensorboard_dir partner_tb/
python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 123 --tensorboard_dir partner_tb/ --order --leave_one_out
python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 720 --tensorboard_dir partner_tb/ --order --leave_one_out

log "context size 1 -- seed 333, 444, 555"
#python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 333 --tensorboard_dir partner_tb/
python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 444 --tensorboard_dir partner_tb/ --order --leave_one_out
#python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 555 --tensorboard_dir partner_tb/

log "context size 1 -- seed 981"
python3 run_baseline_description.py --context_size 1 --load_descriptions --dataset_basedir bangor_data/feature_spk_new_raw --batch_size 16 --self_explain_ngram --only_lil --lamda 0.001 --lamda_spk 0.01 --balanced --max_epochs 10 --gpus 1 --partner --lr 5e-5 --weight_decay 0.001 --seed 981 --tensorboard_dir partner_tb/ --order --leave_one_out