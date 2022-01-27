import argparse
import os
import json
import torch
from load_data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import DistMult, ComplEx, ConvE, TuckER


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--model', default='tucker', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--input', default='numeric', help='Please provide an input path')


def populate_estimate(model, test, suffix):
    ent, rel = [], []
    for i, row in test.iterrows():
        ent.append(ent2idx[row[0]])
        rel.append(rel2idx["Interval-" + row[1] + suffix])
    res = model(torch.LongTensor(ent), torch.LongTensor(rel))

    ind = test.columns.tolist().index(suffix)
    skips = 0
    for i, row in tqdm(test.iterrows()):
        try:
            candidates = list(filter(
                lambda x: row[1] in x,
                list(map(lambda x: idx2ent[x], [x.item() for x in torch.argsort(res[i], descending=True)[:50]]))
            ))
            test.iloc[i, ind] = medians[candidates[0]]
        except IndexError:
            skips += 1
            test.iloc[i, ind] = medians[row[1]]
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
    target = f'{args.input}/{args.dataset}'

    with open(f'{target}/{args.model}_entities.dict') as fd:
        ent2idx = json.load(fd)
    with open(f'{target}/{args.model}_relations.dict') as fd:
        rel2idx = json.load(fd)
    idx2ent = {v: k for k, v in ent2idx.items()}
    with open(f'{target}/medians.dict') as fd:
        medians = json.load(fd)

    d = Data(data_dir=f"{target}/", reverse=True)
    if args.model == "distmult":
        model = DistMult(d, 200, 200, **{"input_dropout": 0.2})
    elif args.model == "complex":
        model = ComplEx(d, 400, 400, **{"input_dropout": 0.2})
    elif args.model == "conve":
        model = ConvE(d, 200, 200, **{"input_dropout": 0.2,
                                      "hidden_dropout1": 0.3,
                                      "feature_map_dropout": 0.2,
                                      "use_bias": True,
                                      "hidden_size": 9728,
                                      "embedding_shape1": 20})
    elif args.model == "tucker":
        if "fb15k237" in args.dataset:
            model = TuckER(d, 200, 200, **{"input_dropout": 0.2,
                                           "hidden_dropout1": 0.4,
                                           "hidden_dropout2": 0.5,
                                           "device": device})
        else:
            model = TuckER(d, 200, 200, **{"input_dropout": 0.2,
                                           "hidden_dropout1": 0.2,
                                           "hidden_dropout2": 0.3,
                                           "device": device})
    else:
        print("Unsupported Model", args.model)
        exit()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{target}/{args.model}.model"))
    else:
        model.load_state_dict(torch.load(f"{target}/{args.model}.model", map_location='cpu'))

    model.eval()

    if 'QOC' in args.dataset or 'FOC' in args.dataset:
        runs = ["_left", "_right"]
    else:
        exit()

    test = pd.read_csv(f'{target}/test_raw.txt', sep='\t', header=None)
    test.columns = ["node", "label", "value"]
    for suffix in runs:
        test[suffix] = np.nan
        populate_estimate(model, test, suffix)

    test["estimate"] = 0
    for suffix in runs:
        test["estimate"] += test[suffix]
    test["estimate"] /= len(runs)

    compute_result(test)
