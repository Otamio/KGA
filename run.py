import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Graph Embedding Algorithms"
)
parser.add_argument('--model', default='rotate', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--options', default='', help='Please provide additional instructions if necessary')
parser.add_argument("--input", type=str, default="data", help="input path")
parser.add_argument("--output", type=str, default="out", help="output path")
parser.add_argument("--use_pbg", action="store_true", help="Use PyTorch-BigGraph to compute embedding")
parser.add_argument("--eval_only", action="store_true", help="Whether to evaluate the current model only")
parser.add_argument("--workers", type=int, default=24, help="Set the number of workers for PBG")
parser.add_argument("--save_best", action="store_true", help="Store the best embedding among evaluated epochs")

if __name__ == "__main__":

    args = parser.parse_args()

    if not args.use_pbg:
        model = args.model
        dataset = args.dataset
        gpu = args.gpu

        if model.startswith("tucker"):
            if "fb15k237" in dataset:
                command = f"CUDA_VISIBLE_DEVICES={gpu} python tucker/main.py --dataset {dataset} --model {model} " \
                          "--num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 " \
                          "--input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1 " \
                          f"--input {args.input} --output {args.output}"
            else:
                command = f"CUDA_VISIBLE_DEVICES={gpu} python tucker/main.py --dataset {dataset} --model {model} " \
                          "--num_iterations 500 --batch_size 128 --lr 0.003 --dr 0.99 --edim 200 --rdim 200 " \
                          "--input_dropout 0.2 --hidden_dropout1 0.2 --hidden_dropout2 0.3 --label_smoothing 0.0 " \
                          f"--input {args.input} --output {args.output}"
        elif model.startswith("distmult"):
            command = f"CUDA_VISIBLE_DEVICES={gpu} python tucker/main.py --dataset {dataset} --model {model} " \
                      "--num_iterations 200 --eval_step 5 --batch_size 128 --lr 0.003 --dr 0.995 " \
                      "--edim 200 --rdim 200 --input_dropout 0.2 --label_smoothing 0.1 " \
                      f"--input {args.input} --output {args.output}"
        elif model.startswith("complex"):
            command = f"CUDA_VISIBLE_DEVICES={gpu} python tucker/main.py --dataset {dataset} --model {model} " \
                      "--num_iterations 200 --eval_step 5 --batch_size 128 --lr 0.003 --dr 0.995 " \
                      "--edim 400 --rdim 400 --input_dropout 0.2 --label_smoothing 0.1 " \
                      f"--input {args.input} --output {args.output}"
        elif model.startswith("conve"):
            command = f"CUDA_VISIBLE_DEVICES={gpu} python tucker/main.py --dataset {dataset} --model {model} " \
                      "--num_iterations 1000 --eval_step 10 --batch_size 128 --lr 0.003 --dr 0.995 " \
                      "--edim 200 --rdim 200 --input_dropout 0.2 --hidden_dropout1 0.3 --feature_map_dropout 0.2 " \
                      f"--label_smoothing 0.1 --use_bias --input {args.input} --output {args.output}"
        elif model.startswith("transe"):
            command = f"python rotate/create_mapping.py --dataset {args.input}/{dataset} && " \
                      f"CUDA_VISIBLE_DEVICES={gpu} python -u rotate/run.py --do_train --cuda --do_valid --do_test " \
                      f"--data_path {args.input}/{dataset} --model {model} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                      f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save {args.output}/{dataset}_{model} " \
                      "--test_batch_size 16"
        elif model.startswith("rotate"):
            command = f"python rotate/create_mapping.py --dataset {args.input}/{dataset} && " \
                      f"CUDA_VISIBLE_DEVICES={gpu} python -u rotate/run.py --do_train --cuda --do_valid --do_test " \
                      f"--data_path {args.input}/{dataset} --model {model} -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv " \
                      f"-lr 0.0001 --max_steps 150000 --valid_steps 5000 -save {args.output}/{dataset}_{model} " \
                      "--test_batch_size 16 -de"
        else:
            print(model, "is not supported")
            exit()

    else:
        command = f"python pbg/main.py --model {args.model} --dataset {args.dataset} " \
                  f"--input {args.input} --output {args.output} --workers {args.workers}"
        if args.eval_only:
            command += " --eval_only"

    if args.options == "dry-run":
        print(command)
    else:
        if args.save_best:
            command += " --save_best"
        subprocess.run(command, shell=True)
