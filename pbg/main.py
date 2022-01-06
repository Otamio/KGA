#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import attr
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.filtered_eval import FilteredRankingEvaluator
from torchbiggraph.train import train
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
)

import os

# remove the Issue: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Model mapping
op_dict = {'complex': "complex_diagonal",
           'distmult': "diagonal",
           'transe': "translation"}

comp_dict = {'complex': "dot",
             'distmult': "dot",
             'transe': "l2"}


def try_to_make_dir(folder):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

######
# Read the parameters
######


parser = argparse.ArgumentParser(
    description="Running PyTorch-BigGraph"
)
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--workers', default=24, help='Please provide the number of CPUs to run the job')
parser.add_argument('--model', default='transe', help='Please provide the model to use')
args = parser.parse_args()

dataset = args.dataset
dim, epochs = 100, 50
method = args.model
workers = args.workers

######
# Adjust different method
######

dataset_path = f"data/{dataset}"
edges_path = f"data/{dataset}/edge"
model_path = f"out/{dataset}/{args.model}"


def get_torchbiggraph_config():

    try_to_make_dir(f"out/{dataset}")
    try_to_make_dir(edges_path)
    try_to_make_dir(model_path)

    config = dict(  # noqa
        # I/O data
        entity_path=f"{edges_path}/entities",
        edge_paths=[
            f"{edges_path}/train",
            f"{edges_path}/valid",
            f"{edges_path}/test",
        ],
        checkpoint_path=model_path,
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": op_dict[method]
            }
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=dim,
        global_emb=False,
        comparator=comp_dict[method],
        # Training
        batch_size=50000,
        num_epochs=epochs,
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        # relation_lr=0.01,
        regularization_coef=1e-3,
        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config


FILENAMES = [
    f"{dataset_path}/train.txt",
    f"{dataset_path}/valid.txt",
    f"{dataset_path}/test.txt"
]


def main():
    config = parse_config(get_torchbiggraph_config())
    if workers > 0:
        config = attr.evolve(config, workers=workers)
        config = attr.evolve(config, batch_size=10000)
        config = attr.evolve(config, num_batch_negs=500)
        config = attr.evolve(config, num_uniform_negs=500)
    elif torch.cuda.is_available():
        config = attr.evolve(config, num_gpus=2)
    else:
        exit()

    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    input_edge_paths = [Path(name) for name in FILENAMES]
    output_train_path, output_valid_path, output_test_path = config.edge_paths

    try:
        os.mkdir(model_path)
    except FileExistsError:
        pass

    log_file_path = f"{model_path}/{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG, filename=str(log_file_path), filemode='w')

    logging.info(f"Set Dataset to {dataset}")
    logging.info(f"Set Model to {args.model}")
    logging.info(f"Set Dimension to {dim}")

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    setup_logging()

    relations = [attr.evolve(r, all_negs=True) for r in config.relations]
    train_config = attr.evolve(config, edge_paths=[output_train_path])
    valid_config = attr.evolve(config, edge_paths=[output_valid_path], relations=relations,
                               num_uniform_negs=0, num_batch_negs=0)
    test_config = attr.evolve(config, edge_paths=[output_test_path], relations=relations,
                              num_uniform_negs=0, num_batch_negs=0)

    # Do the Training here
    train(train_config, evaluator=None, subprocess_init=subprocess_init,
          valid_config=valid_config, test_config=test_config)
    # validation_config = None, numeric_eval_config = None)
    # do_eval(valid_config, evaluator=evaluator, subprocess_init=subprocess_init)
    # do_eval(test_config, evaluator=evaluator, subprocess_init=subprocess_init)

    '''
    do_eval(valid_config, 
            evaluator=FilteredRankingEvaluator(valid_config, filter_paths),
            subprocess_init=subprocess_init
           )

    do_eval(test_config, 
            evaluator=FilteredRankingEvaluator(test_config, filter_paths),
            subprocess_init=subprocess_init
           )
    '''


if __name__ == "__main__":
    main()
