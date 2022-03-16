# Switch-and-Explain
An XLM-RoBERTa based classifier for predicting code-switch points from English-Spanish human--human dialogue.



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
