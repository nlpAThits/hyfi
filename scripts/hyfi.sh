#!/usr/bin/env bash

set -o errexit

# Could be an absolute path to anywhere else
DATA=data

# Dataset
dataset_dir=${DATA}/release
word_embeddings=${DATA}/word-embeds/embeddings.txt

# Checkpoints and prep
prep=${DATA}/prep

do_what=$1
prep_run=$2
this_prep=${prep}/${prep_run}

# to export statistics and the model's weights
mkdir -p ${DATA}
mkdir -p tensorboard
mkdir -p models


if [ "${do_what}" == "get_data" ];
then
    printf "\nDownloading corpus...`date`\n"
    if [ -d "${dataset_dir}" ]; then
        echo "Seems that you already have the dataset!"
    else
        wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz -O ${DATA}/ultrafined.tar.gz
        (cd ${DATA} && tar -zxvf ultrafined.tar.gz && rm ultrafined.tar.gz)
    fi

    printf "\nDownloading word embeddings...`date`\n"
    if [ -d "${DATA}/word-embeds" ]; then
        echo "Seems that you already have the embeddings!"
    else
        mkdir -p ${DATA}/word-embeds
        POINCARE_FILE=poincare_glove_100D_cosh-dist-sq_init_trick.txt
        wget https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn/download?path=%2F\&files=POINCARE_FILE -O ${DATA}/word-embeds/embeddings.txt
    fi

elif [ "${do_what}" == "preprocess" ];
then
    mkdir -p ${this_prep}
    python -u ./preprocess.py \
        --dataset=${dataset_dir} \
        --word2vec=${word_embeddings} \
        --save_data=${this_prep}

elif [ "${do_what}" == "train" ];
then
    python -u ./train.py \
        --data=${this_prep} \
        --export_path=${prep_run}
fi
