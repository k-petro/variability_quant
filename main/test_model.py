#! /home/ec2-user/anaconda3/envs/var_quant_kostas/bin/python3

import argparse
from sklearn.model_selection import KFold, train_test_split
from web_dataloader import web_dataloader
from test_mc_dropout import test_mc_dropout
from test_ngboost import test_ngboost

seed= 42
model_to_test ={
    "mc":test_mc_dropout,
    "ngboost":test_ngboost
}


# User Input
parser=argparse.ArgumentParser()

parser.add_argument('--dataset_name', '-data', required=True, help='Name of the UCI Dataset directory')
parser.add_argument('--n_folds','-nf', default=20, type=int, help='Number of folds to average testing results')
parser.add_argument('--model','-m', required=True, help='Model to be tested')
parser.add_argument('--alpha', '-a', required=True, type=float, help='Parameter alpha for the confidence intervals')

args=parser.parse_args()

dataset_name= args.dataset_name
n_folds= args.n_folds
model = args.model
alpha= args.alpha

# # Iterate all datasets
# for dataset_name in web_dataloader["datasets"][:-1]:

# Load Dataset
dataset = web_dataloader[dataset_name]() #[args.dataset]()
X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values

# Get row indices for the K different dataset splits
folds = KFold(n_splits=n_folds, shuffle=False, random_state=seed).split(X)

# Log results in .txt file
with open("results_"+model+"/"+dataset_name+".csv", "w") as myfile:
    myfile.write("coverage,avg_range,test_ll\n")

# For each dataset split get train, validation, testing
for fold_count, (train_index, test_index) in enumerate(folds):
    print(f"\nStarting fold {fold_count+1} for dataset {dataset_name}.\n")

    X_trainall, X_test = X[train_index], X[test_index]
    y_trainall, y_test = y[train_index], y[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainall, y_trainall, test_size=0.2
    )       

    # Call testing function
    coverage, avg_range, test_ll=\
        model_to_test[model](X_train, y_train, X_val, y_val, X_test, y_test, alpha)

    # Write result
    with open("results_"+model+"/"+dataset_name+".csv", "a") as myfile:
        myfile.write(repr(coverage)+","+repr(avg_range)+","+repr(test_ll)+"\n")
