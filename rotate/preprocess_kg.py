import argparse
import numpy as np
import pandas as pd
import tqdm
import os


parser = argparse.ArgumentParser(
    description='KG preprocessor'
)
parser.add_argument('--dataset', default='fb15k237',
                    help='Please provide a dataset')


def load_data(file_path, ent2idx, rel2idx):
    df = pd.read_csv(file_path, sep='\t', header=None)

    M = df.shape[0]  # dataset size

    X = np.zeros([M, 3], dtype=int)

    for i, row in tqdm.tqdm(df.iterrows()):
        X[i, 0] = ent2idx[row[0]]
        X[i, 1] = rel2idx[row[1]]
        X[i, 2] = ent2idx[row[2]]

    return X


if __name__ == '__main__':

    args = parser.parse_args()
    data_path = args.dataset

    with open(os.path.join(data_path, 'entities.dict')) as fin:
        ent2idx = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            ent2idx[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        rel2idx = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            rel2idx[relation] = int(rid)

    train_path = '{}/train.txt'.format(data_path)
    val_path = '{}/valid.txt'.format(data_path)
    test_path = '{}/test.txt'.format(data_path)

    X_train = load_data(train_path, ent2idx, rel2idx).astype(np.int32)
    X_val = load_data(val_path, ent2idx, rel2idx).astype(np.int32)
    X_test = load_data(test_path, ent2idx, rel2idx).astype(np.int32)

    np.save('{}/train.npy'.format(data_path), X_train)
    np.save('{}/val.npy'.format(data_path), X_val)
    np.save('{}/test.npy'.format(data_path), X_test)
