import argparse
import glob
from collections import namedtuple
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='')
parser.add_argument("--model", type=str, default='')
parser.add_argument("--mode", type=str, default='')
parser.add_argument("--path", type=str, default='out')
Result = namedtuple("Result", "mr mrr hits_1 hits_3 hits_10")


class EpocResult(namedtuple("EpocResult", "epoc validation test")):
    def __gt__(self, other):
        return self.validation.mrr > other.validation.mrr

    def __lt__(self, other):
        return self.validation.mrr < other.validation.mrr

    def __eq__(self, other):
        return self.validation.mrr == other.validation.mrr


def convert_result(dic):
    return Result(dic['mr'],
                  dic['mrr'],
                  dic['hits@1'],
                  dic['hits@3'],
                  dic['hits@10'])


def parse_tucker(args):
    results = {}
    for fname in glob.iglob(f"{args.path}/*{args.dataset}*{args.mode}*{args.model}*.log"):
        try:
            with open(fname) as fd:
                epoc = 0
                result = []
                valid, test = {}, {}
                current = valid
                for line in fd:
                    line = line.split('INFO')[1].strip()
                    if line.startswith("Validation at"):
                        if len(test) > 0:
                            result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
                            valid, test = {}, {}
                        current = valid
                        epoc = int(line.split()[-1])
                    if line.startswith("Test at"):
                        current = test
                    elif line.startswith("Hits @10"):
                        current["hits@10"] = float(line.split(':')[1])
                    elif line.startswith("Hits @3"):
                        current["hits@3"] = float(line.split(':')[1])
                    elif line.startswith("Hits @1"):
                        current["hits@1"] = float(line.split(':')[1])
                    elif line.startswith("Mean rank"):
                        current["mr"] = float(line.split(':')[1])
                    elif line.startswith("Mean reciprocal rank"):
                        current["mrr"] = float(line.split(':')[1])
                result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
            results[fname.split('/')[1].split('.')[0].strip()] = result
        except KeyError as e:
            print('Running:', fname, e)
            pass
    return results


def parse_rotate(args):
    results = {}
    for fname in glob.iglob(f"{args.path}/*{args.dataset}*{args.mode}*{args.model}*/train.log"):
        with open(fname) as fd:
            epoc = 0
            result = []
            valid, test = {}, {}
            for line in fd:
                line = line.strip().split("INFO")[1].strip()
                if len(test) > 0 and line.startswith("Valid MRR at step"):
                    result.append(EpocResult(epoc,
                                             convert_result(valid),
                                             convert_result(test)))
                    valid, test = {}, {}
                    epoc = int(line.split(':')[0].split()[-1])
                if line.startswith("Valid HITS@10 at step"):
                    valid["hits@10"] = float(line.split(':')[1])
                elif line.startswith("Valid HITS@3 at step"):
                    valid["hits@3"] = float(line.split(':')[1])
                elif line.startswith("Valid HITS@1 at step"):
                    valid["hits@1"] = float(line.split(':')[1])
                elif line.startswith("Valid MR at step"):
                    valid["mr"] = float(line.split(':')[1])
                elif line.startswith("Valid MRR at step"):
                    valid["mrr"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@10 at step"):
                    test["hits@10"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@3 at step"):
                    test["hits@3"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@1 at step"):
                    test["hits@1"] = float(line.split(':')[1])
                elif line.startswith("Test MR at step"):
                    test["mr"] = float(line.split(':')[1])
                elif line.startswith("Test MRR at step"):
                    test["mrr"] = float(line.split(':')[1])
            try:
                result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
            except KeyError as e:
                print('Running:', fname, e)
                pass
        exp_name = fname.split('/')[1].split('.')[0].strip()
        results[exp_name] = result
    return results


def parse_pbg(args):
    results = {}
    for fname in glob.iglob(f"{args.path}/*{args.dataset}*/{args.model}*/{args.model}*.log"):
        try:
            with open(fname) as fd:
                epoc = 0
                result = []
                valid, test = {}, {}
                for line in fd:
                    try:
                        line = line.split('INFO:')[1].strip()
                        if len(line) == 0:
                            continue
                    except IndexError:
                        continue
                    if line.startswith("Finished epoch"):
                        epoc = int(line.split('Finished epoch')[1].split('/')[0])
                    if line.startswith("Stats: loss:"):
                        line = line[6:].strip()
                        values = list(map(lambda x: float(x.split(':')[1]), line.split(',')))
                        if len(test) > 0:
                            result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
                            valid, test = {}, {}
                        if len(valid) == 0:
                            _, valid["mr"], valid["mrr"], valid["hits@1"], valid["hits@10"], _, _, _ = values
                            valid["hits@3"] = 0.0
                        else:
                            _, test["mr"], test["mrr"], test["hits@1"], test["hits@10"], _, _, _ = values
                            test["hits@3"] = 0.0
                result.append(EpocResult(epoc, convert_result(valid), convert_result(test)))
            results[fname.split('/')[1].split('.')[0].strip() + '_' + fname.split('/')[-2]] = result
        except KeyError as e:
            print("Error:", fname, e)
            pass
    return results


def main():

    args = parser.parse_args()
    if 'DWD' not in args.dataset:
        results_rotate = parse_rotate(args)
        results_tucker = parse_tucker(args)
        results_pbg = {}
    else:
        results_rotate = {}
        results_tucker = {}
        results_pbg = parse_pbg(args)

    frame = []
    for idx, result in enumerate([results_rotate, results_tucker, results_pbg]):
        for exp, res in sorted(result.items()):
            try:
                best = max(res)
                exp, model = exp.rsplit('_', 1)
                frame.append({
                    "dataset": exp,
                    "model": model,
                    "best_epoc": best.epoc // 100 if idx == 0 else best.epoc,
                    "mrr": round(best.test.mrr, 3),
                    "hits@1": round(best.test.hits_1, 3),
                    "hits@10": round(best.test.hits_10, 3)
                })
            except ValueError:
                print("Cannot process out:", exp)

    if len(frame) > 0:
        print(pd.DataFrame(frame).sort_values(by=["model", "mrr", "hits@1", "hits@10"], ascending=False))


if __name__ == "__main__":
    main()
