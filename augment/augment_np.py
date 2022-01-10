from augment_main import augment_np
from loader import get_data_np
import argparse
from constant import *
import os

parser = argparse.ArgumentParser(
    description='Create literals and append to graph'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
parser.add_argument('--mode', default='All', metavar='',
                    help='which augmentation mode to run?')
parser.add_argument('--bins', default='8', metavar='',
                    help='How many bins to run?')
parser.add_argument('--reverse', action="store_true",
                    help='Whether to create reverse relations in the dataset?')
args = parser.parse_args()

# Read configuration
home = os.environ['HOME']
dataset = args.dataset
modes = args.mode.split(',')
bins = int(args.bins)

if modes[0] == "All":
    modes = SUPPORTED_MODE

# Get data
entities, train, valid, test = get_data_np(dataset)

for mode in modes:
    if mode in SUPPORTED_MODE:
        augment_np(entities, train, valid, test,
                   dataset, mode, bins, reverse=args.reverse)
