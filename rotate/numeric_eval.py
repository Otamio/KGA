import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import KGEModel
from dataloader import TestDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--model', default='rotate', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')

mapping = {
    "transe": "TransE",
    "rotate": "RotatE"
}


def populate_estimate(model, test, suffix):
    ent_rel = []
    for i, row in test.iterrows():
        ent_rel.append([ent2idx[row[0]],
                        rel2idx["Interval-" + row[1] + suffix],
                        0])
    preds = []
    skips = 0
    for s in tqdm(range(0, len(ent_rel), 16)):
        sli = ent_rel[s:s+16]
        res = model((torch.LongTensor(sli).view(-1, 3),
                     torch.LongTensor([range(len(ent2idx))])),
                    mode="tail-batch")
        for i in range(min(16, len(sli))):
            candidates = list(filter(
                lambda x: test.iloc[s+i, 1] in x,
                [idx2ent[x.item()] for x in torch.argsort(res[i], descending=True)[:50]]
            ))
            try:
                preds.append(medians[candidates[0]])
            except IndexError:
                skips += 1
                preds.append(medians[test.iloc[s+i, 1]])

    ind = test.columns.tolist().index(suffix)
    for i in range(len(test)):
        test.iloc[i, ind] = preds[i]
    print('Skipped', skips)


def compute_result(test):
    res = []
    test['MAE'] = abs(test['estimate'] - test['value'])
    for p in test['label'].unique():
        sli = test[test['label'] == p]
        res.append({
            "Property": p,
            "MAE": round(sli['MAE'].mean(), 3 if 'height' in p else 1)
        })
    print(pd.DataFrame(res))


if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    target_src = f'numeric/{args.dataset}'
    target_dst = f'numeric/{args.dataset}_{args.model}'

    with open(os.path.join(target_src, 'entities.dict')) as fin:
        ent2idx = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            ent2idx[entity] = int(eid)

    with open(os.path.join(target_src, 'relations.dict')) as fin:
        rel2idx = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            rel2idx[relation] = int(rid)

    idx2ent = {v: k for k, v in ent2idx.items()}
    with open(f'{target_src}/medians.dict') as fd:
        medians = json.load(fd)

    if args.model == "transe":
        model = KGEModel(model_name=mapping[args.model], nentity=len(ent2idx), nrelation=len(rel2idx),
                         hidden_dim=1000, gamma=24.0, double_entity_embedding=False,
                         double_relation_embedding=False)
    elif args.model == "rotate":
        model = KGEModel(model_name=mapping[args.model], nentity=len(ent2idx), nrelation=len(rel2idx),
                         hidden_dim=1000, gamma=24.0, double_entity_embedding=True,
                         double_relation_embedding=False)
    else:
        exit()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{target_dst}/{mapping[args.model]}.model"))
    else:
        model.load_state_dict(torch.load(f"{target_dst}/{mapping[args.model]}.model", map_location='cpu'))

    model.eval()

    if 'QOC' in args.dataset or 'FOC' in args.dataset:
        runs = ["_left", "_right"]
    else:
        exit()

    test = pd.read_csv(f'{target_src}/test_raw.txt', sep='\t', header=None)
    test.columns = ["node", "label", "value"]
    for suffix in runs:
        test[suffix] = np.nan
        populate_estimate(model, test, suffix)

    test["estimate"] = 0
    for suffix in runs:
        test["estimate"] += test[suffix]
    test["estimate"] /= len(runs)

    compute_result(test)
