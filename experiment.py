"""
Experiment implementation for MSR Mining Challenge 2021
- Compares 2 Random Forest classifier models trained with and without file depth
- Output: Feature importance weighting and model accuracy scores for each model for comparison
"""

import json
import argparse
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler


def run_experiment(data, output_file):
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
    print(df.head())  # For debug/ref: print first 5 data entries from JSON to check format is valid

    # Extract relevant feature columns i.e. numerical fields
    cols = ["fixLineNum", "fixNodeLength", "fixNodeStartChar", "bugNodeLength", "bugNodeStartChar", "bugLineNum",
            "fileDepthNumber"]
    y = df["bugType"]  # y = target variable (bugType)

    # Undersampling: Quick and dirty method = Select N random samples from largest class matching size of smallest class
    # TODO: Implement seeding for reproduceability?
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(df, y)
    X_under['fileDepthNumber'] = X_under['bugFilePath'].str.count("/")

    # Split data to 80:20 train:test
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2)

    # TODO: Implement feature selection method for creating control model. Else: pre-select based on analysis
    # Use feature selection to choose from 'cols' features which would give good accuracy to a model
    # Preliminary manual analysis suggests ["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "fixNodeStartChar"]
    
    X_train_1 = X_train[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "bugLineNum"]]  # X = the feature set
    X_test_1 = X_test[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "bugLineNum"]]

    #X_train_1.to_json(r'x_train_1.json')
    #X_test_1.to_json(r'x_test_1.json')
    
    # feature_model = RandomForestClassifier(random_state=100, n_estimators=50)
    # feature_model.fit(X_train_1, y_train)
    # print("Feature selection modelling - Feature importances: \n" + feature_model.feature_importances_)  # for debug
    #
    # sel_model_tree = SelectFromModel(estimator=feature_model, prefit=True, threshold='mean')
    # X_train_sfm_tree = sel_model_tree.transform(X_train_1)
    # print(sel_model_tree.get_support())

    # Model 1: Control model - does not include fileDepthNumber feature
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_1, y_train)
    y_pred = clf.predict(X_test_1)

    # Model 2: Variation model of control with fileDepthNumber feature included

    X_train_2 = X_train[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "bugLineNum", "fileDepthNumber"]]
    X_test_2 = X_test[["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "bugLineNum", "fileDepthNumber"]]

    #X_train_2.to_json(r'x_train_2.json')
    #X_test_2.to_json(r'x_test_2.json')

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
    print("(+ fileDepthNumber) Feature importances \n")
    print(str(importance2))
    print("(+ fileDepthNumber) Model accuracy: " + str(acc2))

    # Write results to output file
    with open(output_file, 'w', encoding="utf8") as f:
        f.write("(Control) Feature importances \n")
        f.write(str(importance1))
        f.write("\n(Control) Model accuracy: " + str(acc))
        f.write("\n\n(+ fileDepthNumber) Feature importances \n")
        f.write(str(importance2))
        f.write("\n(+ fileDepthNumber) Model accuracy: " + str(acc2))
        f.close()

    # TODO: Output to CSV for easier results processing?
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
        default='sstubs.json',
        type=str,
        help='Path to data file (JSON)'
    )
    parser.add_argument(
        '--output',
        default='results.txt',
        type=str,
        help='Path/name of output file to write results out to.'
    )
    args = parser.parse_args()

    # Run experiment function using console input parameters
    run_experiment(args.data, args.output)


if __name__ == '__main__':
    main()
