from __future__ import print_function

import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix


import xgboost as xgb
from xgboost.sklearn import XGBClassifier # how do we make sure we are using this
from sklearn import metrics   #Additional scklearn functions

def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = y_train.unique()
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]

    return sample_weights


def input_fn(request_body, request_content_type):
    """An input_fn that loads a numpy array"""
    if request_content_type == "text/csv":

        input_features =[]
        for i in request_body.split('\n'): # the first element is the id, the rest is payload            
            if len(i) == 0: continue
            input_features.append([float(j) for j in i.split(",")])
        return np.array(input_features)
    else:
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--n_estimators', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=4)    
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--objective', type=str, default='multi:softmax')
    parser.add_argument('--subsample', type=float, default=1)
    parser.add_argument('--reg_lambda', type=float, default=0.1)
    parser.add_argument('--eval_metric', type=str, default='merror') #- looks like we don't include this in fact, worth checking later
    parser.add_argument('--colsample_bytree', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)


    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files1 = [ os.path.join(args.train, file1) for file1 in os.listdir(args.train) ]
    if len(input_files1) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train))
    raw_data1 = [pd.read_csv(file1, header=None, engine='python') for file1 in input_files1]
    training_data = pd.concat(raw_data1)

    input_files2 = [ os.path.join(args.validation, file2) for file2 in os.listdir(args.validation) ]
    if len(input_files2) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.validation))
    raw_data2 = [pd.read_csv(file2, header=None, engine='python') for file2 in input_files2]
    validing_data = pd.concat(raw_data2)

    y_train = training_data.iloc[:,0].values
    X_train = training_data.iloc[:,1:].values
    
    y_test = validing_data.iloc[:,0].values
    X_test = validing_data.iloc[:,1:].values

    largest_class_weight_coef = max(training_data.iloc[:,0].value_counts().values)/training_data.shape[0]
    w_train = CreateBalancedSampleWeights(training_data.iloc[:,0], largest_class_weight_coef=largest_class_weight_coef)


    clf = xgb.XGBClassifier(n_estimators=args.n_estimators, 
    n_jobs=args.n_jobs, 
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    subsample=args.subsample,
    objective=args.objective)

    clf = clf.fit(X_train, y_train, 
    sample_weight=w_train, 
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='merror',
    verbose=True)

    evals_result = clf.evals_result()

    # save the model
    model_location = args.model_dir + '/xgboost-model'
    pkl.dump(clf, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))


def model_fn(model_dir):
    """Deserialize and return fitted model.
    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    clf = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return clf
