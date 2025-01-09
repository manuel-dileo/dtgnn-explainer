# Explainability framework for discrete-time graph neural networks
This repository contains the code behind the paper UNDER REVIEW

## Supplementary Material
The supplementary material is available as a pdf in this repo.

## Reproducibility
To reproduce our experiments, proceed with the following steps:
- Create the log directories using the `create_log_dir.sh` script
 ```
 chmod +x ./create_log_dir.sh
 ./create_log_dir.sh
 ```
- Follow the steps in the `Prepare the datasets` section of this README
- Train the base-models (optional). The weights of the well-trained base-models are already available for all the considered datasets in folder `trained_models`. Hence, it is not necessary to re-train the base-models. However, if you want to do it, simply run
 ```
 python train.py --dataset ${datasetname} --model ${modelname}
 ```
- Train the explainers and evaluate their temporal fidelity. Simply run:
  ```
  python explain_fid.py --dataset ${datasetname} --model ${modelname} --xai_model ${xainame}
  ```
  where `xai_model` is a choice between ['gnnexplainer', 'last', 'khop', 'sa', 'ig', 'dummy', 'pg']. Check the help for the `--xai_model` option for more information.
- Train the explainers and evaluate their human readability level (cohesivennes, edge recurrence, reciprocity, homophily). Simply run:
  ```
  python explain_eval.py --dataset ${datasetname} --model ${modelname} --xai_model ${xainame}
  ```

## Prepare the datasets
- Create a directory `dataset`
- Get `reddit-title.tsv` from [Roland repository](https://github.com/snap-stanford/roland). Place it in a directory called `roland_public_data`
- Download `email-Eu-core-temporal-Dept1.txt.gz` from [SNAP](https://snap.stanford.edu/data/email-Eu-core-temporal.html). Unzip it, rename the txt file as `email-eu.txt`, and place it in a folder called `email-eu`.

## Run on your own dataset
To run the training of the base-models, and the evaluation of the explainability technique on your link prediction task, just simply start from adding a choice with your dataset name in the `--dataset` options. Then, follow the errors the code will generate to add where needed all the options/code to handle your dataset(e.g. the repo has several dispatch tables with keys equals to the dataset names). Keep in mind that a discrete-time temporal network is represented as a list of PyG Data object in chronologic order, each representing a specific snapshot.

## Contact information
For any comment, suggestion or question, please do not hesitate to contact me at name dot surname at institute dot nation