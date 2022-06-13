# Switch-and-Explain
An XLM-RoBERTa based classifier for predicting code-switch points from English-Spanish human--human dialogue.

Link to the full paper and citation [here](https://aclanthology.org/2022.acl-long.267/).


This code incorporates the LIL layer from the SelfExplain framework (https://arxiv.org/abs/2103.12279)

Make sure to unzip the data folder and put it in an enclosing folder named bangor_data, otherwise rename the filepaths for '$DATA_FOLDER' or --dataset_basedir in the sh scripts below.

## Preprocessing

Data for preprocessing available in `bangor_data/` folder.

To run scripts for getting the phrase masks, use:

# For baseline masking:
```shell
sh scripts/run_preprocessing_bangor_idx.sh
```
# For masking speaker descriptions + dialogues (this will take a while)
```shell
sh scripts/run_preprocessing_bangor_desc.sh
```

## Training Baselines on 10 seeds for context size 1, and extracting LIL interpretations:

```shell
sh scripts/baseline_train_ctx1.sh
```

## Training speaker List models on 10 seeds for context size 1 and extracting LIL interpretations:
```shell
sh scripts/list_models_ctx1.sh
```

## Update - new files

Preprocessed data is available for download [here](https://drive.google.com/file/d/12w5b2djUr984bJmylciZec3jMu5uG_10/view?usp=sharing).

Data for control experiments is available for download [here](https://drive.google.com/file/d/13oJaFZq-1zD9tLfDgIpl4DoH2DOVLZ4q/view?usp=sharing).

Extract these under the `bangor_data` folder.

Model outputs for the unbalanced validation and test sets is available under `model_outputs`. Folders are organized by split (test or validation), model type (speaker-prompted or baseline), and context size (in number of previous utterances).
