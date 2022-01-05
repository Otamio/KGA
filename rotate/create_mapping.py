import argparse

parser = argparse.ArgumentParser(
    description="Mapping Creator"
)
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset_path = args.dataset
    entities = set()
    relations = set()

    for fname in [f"{dataset_path}/train.txt",
                  f"{dataset_path}/valid.txt",
                  f"{dataset_path}/test.txt"]:
        with open(fname) as fd:
            for line in fd:
                s, r, o = [x.strip() for x in line.split()]
                entities.add(s)
                relations.add(r)
                entities.add(o)

    with open(f"{dataset_path}/entities.dict", 'w') as fd:
        for i, entity in enumerate(entities):
            fd.write(f"{i}\t{entity}\n")

    with open(f"{dataset_path}/relations.dict", 'w') as fd:
        for i, relation in enumerate(relations):
            fd.write(f"{i}\t{relation}\n")
