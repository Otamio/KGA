import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    gpu = args.gpu
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model transe", shell=True)
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model distmult", shell=True)
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model complex", shell=True)
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model conve", shell=True)
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model rotate", shell=True)
    subprocess.run(f"python run.py --dataset {dataset} --gpu {gpu} --model tucker", shell=True)
