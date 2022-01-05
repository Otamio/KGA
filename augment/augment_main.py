from constant import *
from augment_utils import *
from collections import defaultdict
import os
import json
import shutil


def try_to_make_dir(folder):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass


##########################################
#    Main Function
##########################################


def augment_lp(entities, df, dataset, mode, bins=None):
    suffix = int(np.log2(bins)) if mode.endswith("Hierarchy") else bins

    if mode in CHAINABLE_MODE:
        print(f'Running mode {mode}')

        (numeric_edges_processed, _, _), _, qnode_edges = create_new_edges(df, mode, bins)

        # Write the augmented version (without chain)
        target = f'datasets/{dataset}/processed/{dataset}_{mapping_no_chain[mode]}_{suffix}'
        try_to_make_dir(target)
        pd.concat([entities, numeric_edges_processed]) \
            .to_csv(f'{target}/train.txt', sep='\t', header=False, index=False)
        shutil.copy(f'datasets/{dataset}/data/valid.tsv', f'{target}/valid.txt')
        shutil.copy(f'datasets/{dataset}/data/test.tsv', f'{target}/test.txt')

        # Write the augmented version (with chain)
        target = f'datasets/{dataset}/processed/{dataset}_{mapping_chain[mode]}_{suffix}'
        try_to_make_dir(target)
        pd.concat([entities, numeric_edges_processed, pd.DataFrame(qnode_edges)]) \
            .to_csv(f'{target}/train.txt', sep='\t', header=False, index=False)
        shutil.copy(f'datasets/{dataset}/data/valid.tsv', f'{target}/valid.txt')
        shutil.copy(f'datasets/{dataset}/data/test.tsv', f'{target}/test.txt')


def augment_np(entities, train, valid, test, dataset, mode, bins=None):
    suffix = int(np.log2(bins)) if mode.endswith("Hierarchy") else bins

    if mode in CHAINABLE_MODE:

        print(f'Running mode {mode}')

        (train_edges_processed, valid_edges_processed, test_edges_processed), \
            (train_edges_raw, valid_edges_raw, test_edges_raw), qnode_edges = \
            create_new_edges(train, mode, bins, valid=valid, test=test)

        medians_dict = {}
        collections = defaultdict(list)
        collections_raw = defaultdict(list)

        for i, row in train_edges_raw.iterrows():
            collections_raw[row['node1'] + '  ' + row['label']].append(row['node2'])

        for i, row in train_edges_processed.iterrows():
            key = row['node1'] + '  ' + row['label'].split('-')[1].rsplit('_', 1)[0]
            for item in collections_raw[key]:
                collections[row['node2']].append(item)

        for k, v in collections.items():
            medians_dict[k] = np.median(v)

        # Finally, add the median of each property as a baseline
        for property_ in train_edges_raw['label'].unique():
            medians_dict[property_] = \
                train_edges_raw[train_edges_raw['label'] == property_]['node2'].median()

        # Write the original version
        target = f'datasets/{dataset}/numeric/{dataset}_{mapping_no_chain[mode]}_{suffix}'
        try_to_make_dir(target)
        pd.concat([entities, train_edges_processed]) \
            .to_csv(f'{target}/train.txt', sep='\t', header=False, index=False)
        valid_edges_processed.to_csv(f'{target}/valid.txt', sep='\t', header=False, index=False)
        test_edges_processed.to_csv(f'{target}/test.txt', sep='\t', header=False, index=False)
        valid_edges_raw.to_csv(f'{target}/valid_raw.txt', sep='\t', header=False, index=False)
        test_edges_raw.to_csv(f'{target}/test_raw.txt', sep='\t', header=False, index=False)
        with open(f'{target}/medians.dict', 'w+') as fd:
            json.dump(medians_dict, fd, indent=2)

        # Write the chaining version
        target = f'datasets/{dataset}/numeric/{dataset}_{mapping_chain[mode]}_{suffix}'
        try_to_make_dir(target)
        pd.concat([entities, train_edges_processed, pd.DataFrame(qnode_edges)]) \
            .to_csv(f'{target}/train.txt', sep='\t', header=False, index=False)
        valid_edges_processed.to_csv(f'{target}/valid.txt', sep='\t', header=False, index=False)
        test_edges_processed.to_csv(f'{target}/test.txt', sep='\t', header=False, index=False)
        valid_edges_raw.to_csv(f'{target}/valid_raw.txt', sep='\t', header=False, index=False)
        test_edges_raw.to_csv(f'{target}/test_raw.txt', sep='\t', header=False, index=False)
        with open(f'{target}/medians.dict', 'w+') as fd:
            json.dump(medians_dict, fd, indent=2)
