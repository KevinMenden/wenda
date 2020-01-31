"""
Run the standard Wenda pipeline on a dataset

Requires formatted and processed training data and prediction data
"""

import argparse
import os
import sys
import pandas
import numpy as np
import sklearn.linear_model as lm
import feature_models
import GPy
import itertools
import time
import datetime
import data as data_mod
import models
import util

sys.path.append("")


## Argument parsing
parser = argparse.ArgumentParser(description='Run Wenda')
parser.add_argument("--train_data", help="Path to training data")
parser.add_argument("--train_pheno", help="Path to training metadata")
parser.add_argument("--pred_data", help="Path to prediction data")
parser.add_argument("--pred_pheno", help="Path to prediction metadata")
parser.add_argument("-t", "--threads", help="Number of threads to use", default=10, type=int)
parser.add_argument("-f", "--features", help="Feature model dir", default=".")
parser.add_argument("-c", "--confidences", help="Confidence value dir", default=".")
parser.add_argument("-o", "--out", help="Output directory", default=".")
parser.add_argument("-a", "--alpha", help="Alpha value. Default: 0.8", default=0.8)
parser.add_argument("--splits", help="Number of CV splits", default=10)
parser.add_argument("-k", help="K value. Default: 3", default=3)

# parse arguments
args = parser.parse_args()
training_data_path = args.train_data
training_pheno_path = args.train_pheno
pred_data_path = args.pred_data
pred_pheno_path = args.pred_pheno
n_jobs = args.threads
feature_model_dir = args.features
confidences_dir = args.confidences
predict_path = args.out
alpha = args.alpha
n_splits = args.splits
k_wnet = args.k


# Load the data
print("Loading the data ...")
train = data_mod.MethylationDataset.read(training_data_path, training_pheno_path)
pred = data_mod.MethylationDataset.read(pred_data_path, pred_pheno_path)
print("Data loaded.")

# Define normalizers
normalizer_x = util.StandardNormalizer
normalizer_y = util.HorvathNormalizer

# Define feature model params
feature_model_type = feature_models.FeatureGPR
feature_model_params = {"kernel": GPy.kern.Linear(input_dim=train.getNofCpGs()-1)}

# Make grouping variable for target tissue
# This currently only works for one target tissue - code has to be modified for several
# (Or stor it in the pheno data to begin with)
tt = ["target_tissue"] * pred.pheno_table.shape[0]
pred.pheno_table['tissue'] = tt
grouping = pred.pheno_table['tissue']


# Create the Wenda model
model = models.Wenda(x_train=train.meth_matrix,
                    y_train=train.age,
                    x_test=pred.meth_matrix,
                    norm_x=normalizer_x,
                    norm_y=normalizer_y,
                    feature_model_dir=feature_model_dir,
                    feature_model_type=feature_model_type,
                    feature_model_params=feature_model_params,
                    confidences_dir=confidences_dir,
                    n_jobs=n_jobs)


# Fit feature models
print("Fitting feature models ...")
model.fitFeatureModels()

# Collect confidences
print("Collecting confidences ...")
model.collectConfidences()

# Perform training and prediction
print("Performing training and prediction")
k_wnet = 3
weight_func = lambda x: np.power(1-x, k_wnet, alpha=0.8)
predictions = model.predictWithTrainingDataCV(weight_func=weight_funct,
                                              grouping=grouping,
                                              predict_path=predict_path,
                                              alpha=0.8, 
                                              n_splits=10)

print("Finished.")