#!/bin/bash


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
conda activate p36-clean


export DATA_FOLDER='bangor_data/feature_spk_new_random'
export TOKENIZER_NAME='xlm-roberta-base'
export MAX_LENGTH=45

# Creates jsonl files for train and dev

python preprocessing/bangor_lil_idx.py \
      --data_dir $DATA_FOLDER  \
      --tokenizer_name $TOKENIZER_NAME

python preprocessing/bangor_lil_idx.py \
      --data_dir $DATA_FOLDER  \
      --tokenizer_name $TOKENIZER_NAME --balanced

