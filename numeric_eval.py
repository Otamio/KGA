import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Numeric Evaluation"
)
parser.add_argument('--model', default='transe', help='Please provide a model to test')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--input', default='numeric', help='Please provide an input path')


if __name__ == "__main__":

    args = parser.parse_args()
    if args.model in ['transe', 'rotate']:
        subprocess.run(f"python rotate/numeric_eval.py --dataset {args.dataset} --gpu {args.gpu} "
                       f"--model {args.model} --input {args.input}",
                       shell=True)
    else:
        subprocess.run(f"python tucker/numeric_eval.py --dataset {args.dataset} --gpu {args.gpu} "
                       f"--model {args.model} --input {args.input}",
                       shell=True)
