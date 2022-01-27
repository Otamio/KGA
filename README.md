## KGA: Knowledge Graph Augmentation

---

This is the repo for the submission *Augmenting Knowledge Graphs for Better Link Prediction*. This repo is structured as follows:
1. `augment`. This directory contains the code to run the literal graph augmentation for the input graph. All input are tab-separated files. `augment_lp.py` is used to produce graph for link prediction, and `augment_np.py` is used to produce graph for numeric prediction.
   1. To augment the dataset with link prediction, make sure the directory `data/{dataset}` contains at least four files:
      1. `train.txt`: The training entity triples.
      2. `valid.txt`: The validation entity triples.
      3. `test.txt`: The testing entity triples.
      4. `numerical_literals.txt`: The literal triples.
      5. Once you get the above files, augment the graph with `python augment/augment_lp.py --dataset {dataset} --bins {bins}`.
   2. To augment the dataset with numericaprediction, make sure the directory `data/{dataset}` contains at least four files:
      1. `train_kge`: The entity triples.
      2. `train_100`: The training literal triples.
      3. `dev`: The validation literal triples.
      4. `test`: The test literal triples.
      5. Once you get the above files, augment the graph with `python augment/augment_np.py --dataset {dataset} --bins {bins}`.
2. `pbg`. This directory contains the code to run Darpa Wikidata using PyTorch-BigGraph. To run the code, we recommend installing PyTorch-BigGraph as documented in https://github.com/facebookresearch/PyTorch-BigGraph, and Faiss as documented in https://github.com/facebookresearch/faiss.
3. `rotate`. This directory contains the code to run TransE and RotatE with negative sampling. To run the code, we recommend installing the environment documented in https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.
4. `tucker`. This directory contains the code to run DistMult, ComplEx, ConvE, and TuckER with k-N sampling. To run the code, we recommend installing the environment documented in https://github.com/ibalazevic/TuckER.
5. `data`. This directory is the default location to store the input graphs. The data for this project can be located at: https://drive.google.com/drive/folders/14XtfAsfchsS-gPUZ1_YtP1X3bFiCaOS6?usp=sharing. 
6. `out`. This directory is the default location to store output logs.
7. `numeric`. This directory is the default location to store numeric prediction logs. Since augmenting the graph for either link prediction and numeric prediction will produce different data, we recommend using directory `data` to store the augmented graph for link prediction, and directory `numeric` to store the augmented graph for numeric prediction.

The directory also contains the scripts which are deemed as the entry point of run the program:
1. `run.py` is the script to run base embedding models, KBLN, and LiteralE on the given graph. 
   1. To run link prediction on the dataset: `python run.py --dataset {dataset} --model {model}`
   2. To run link prediction on the dataset with PyTorch-BigGraph: `python run.py --dataset {dataset} --model {model} --use_pbg`
   3. Of course, you can change the input folder if you save the augmented graphs in other locations. You can also change the output folder if needed. To run numeric prediction on the dataset: `python run.py --dataset {dataset} --model {model} --input {input_path} --output {output_path}`.
2. `run_batch.py` is the wrapper for run.py if user wants to run all six base models for an input graphs.
3. `summary.py` is the script to collect best results from the logs. Users can  run the program using `python summary.py --dataset {dataset} --model {model}` to get the best metric for a given model. The program will run through all iterations of the log files and print the metrics of the epoch with best `validation MRR`. The results will be ordered by model (if user does not specify a model), mrr, hits@1 and finally hits@10.
4. `summary.py` 