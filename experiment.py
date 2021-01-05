"""
Experiment implementation for MSR Mining Challenge 2021
- Compares 2 Random Forest classifier models trained with and without file depth
- Output: Feature importance weighting and model accuracy scores for each model for comparison
"""

import json
import argparse
import pickle
import csv
import pandas as pd
import os.path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler


def run_experiment(data, output_file, testSize, feature):
    """
    Run the Random Forest model comparison experiment.
    1. Input: Path to data file
    2. Preproc.: Calculate fileDepthNumber
    3. Preproc.: Undersample largest class (randomly cut data points to N where N = number of entries in smallest class)
    4. Build models: Control model without fileDepthNumber, identical model with fileDepthNumber feature included
    5. Write out feature importance and model accuracy scores to file
    """
    
    # Load data from JSON file
    df = pd.read_json(data)
    #print(df.head())  # For debug/ref: print first 5 data entries from JSON to check format is valid

    y = df["bugType"]  # y = target variable (bugType)

    # Undersampling: Quick and dirty method = Select N random samples from largest class matching size of smallest class
    # TODO: Implement seeding for reproduceability?
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(df, y)
    if feature == 'fileDepthNumber':
        X_under['fileDepthNumber'] = X_under['bugFilePath'].str.count("/")

    # Split data to 80:20 train:test
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size= testSize)

    # TODO: Implement feature selection method for creating control model. Else: pre-select based on analysis
    # Use feature selection to choose from 'cols' features which would give good accuracy to a model
    # Preliminary manual analysis suggests ["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "fixNodeStartChar"]
    # SKIPPED due to bugs and time constraint - hard coded pre-selected features for dirty fix
    
    X_train_1 = X_train[["fixNodeLength", "bugNodeLength", "fixNodeStartChar", "bugNodeStartChar"]]  # X = the feature set
    X_test_1 = X_test[["fixNodeLength", "bugNodeLength", "fixNodeStartChar", "bugNodeStartChar"]]
    
    # feature_model = RandomForestClassifier(random_state=100, n_estimators=50)
    # feature_model.fit(X_train_1, y_train)
    # imp = feature_model.feature_importances_
    # print("Feature selection modelling - Feature importances: \n" + imp)  # for debug
    #
    # sel_model_tree = SelectFromModel(estimator=feature_model, prefit=True, threshold='mean')
    # X_train_sfm_tree = sel_model_tree.transform(X_train_1)
    # print(sel_model_tree.get_support())

    # Model 1: Control model - does not include fileDepthNumber feature
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_1, y_train)
    y_pred = clf.predict(X_test_1)

    # Model 2: Variation model of control with fileDepthNumber feature included

    X_train_2 = X_train[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "fixNodeStartChar", feature]]
    X_test_2 = X_test[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "fixNodeStartChar", feature]]

    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(X_train_2, y_train)
    y2_pred = clf2.predict(X_test_2)

    # Gather feature importance and model accuracy scores

    importance1 = pd.Series(clf.feature_importances_, index=X_test_1.columns)
    importance2 = pd.Series(clf2.feature_importances_, index=X_test_2.columns)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc2 = metrics.accuracy_score(y_test, y2_pred)

    # (Debug) Print output to console
    print("(Control) Feature importances \n")
    print(str(importance1))
    print("(Control) Model accuracy: " + str(acc))
    print("(+ " + feature + ") Feature importances \n")
    print(str(importance2))
    print("(+ " + feature + ") Model accuracy: " + str(acc2))

    # Write results to output file
    res = {}
    for index, value in importance1.items():
        res['control_' + index]= str(value)
    res['Control_model_accuracy'] = str(acc)        
    for index, value in importance2.items():
        res[ feature + '_' + index]= str(value) 
    res[ feature + '_model_accuracy'] = str(acc2)

    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='',encoding="utf8") as f:  # You will need 'wb' mode in Python 2.x
        w= csv.DictWriter(f, res.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(res)

    # TODO: Pickle the model(s) afterwards for reproduceability/documentation?


def calculate_file_depth(data):
    # Based on Tiger's script: calculate and append fileDepthNumber to data list
    with open(data, encoding="utf8") as f:
        d = json.load(f)

    datacopy = d.copy()

    for x in datacopy:
        filedepth = x["bugFilePath"].count("/")
        x['fileDepthNumber'] = filedepth
    return datacopy


def main():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument(
        '--data',
        default='sstubsLarge-0104.json',
        type=str,
        help='Path to data file (JSON)'
    )
    parser.add_argument(
        '--output',
        default='results-bugLineNum-0.1.csv',
        type=str,
        help='Path/name of output file to write results out to.'
    )
    parser.add_argument(
        '--feature',
        default='bugLineNum',
        type=str,
        help='Name of feature to test'
    )
    parser.add_argument(
        '--repetitions',
        default=100,
        type=int,
        help='How many times to run the experiment'
    )
    # parser.add_argument(
    #     '--tt_seed',
    #     default=42,
    #     type=int,
    #     help='Random seed for test-train set splitting'
    # )
    # parser.add_argument(
    #     '--us_seed',
    #     default=42,
    #     type=int,
    #     help='Random seed for undersampling of data'
    # )
    parser.add_argument(
        '--testSize',
        default= 0.1,
        type=int,
        help='The test size of the dataset'
    )
    args = parser.parse_args()

    # Run experiment function using console input parameters
    for _ in range(args.repetitions):
        run_experiment(args.data, args.output, args.testSize,args.feature)


if __name__ == '__main__':
    main()
