Please find the data for this project at:
https://drive.google.com/drive/folders/14XtfAsfchsS-gPUZ1_YtP1X3bFiCaOS6?usp=sharing


To augment the dataset:
`python augment/augment_lp.py --dataset {dataset} --bins {bins}`

`python augment/augment_np.py --dataset {dataset} --bins {bins}`

To run link prediction:
`python run.py --dataset {dataset} --model {model}`

To run value imputation:
`python run.py --dataset {dataset} --model {model} --input numeric --output numeric`
