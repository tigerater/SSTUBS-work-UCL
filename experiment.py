"""
Experiment implementation for MSR Mining Challenge 2021
- Compares 2 Random Forest classifier models trained with and without file depth
- Output: Feature importance weighting and model accuracy scores for each model for comparison
"""

import json
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

    # TODO: implement method/code to preprocess data(?) e.g. oversampling and calculate fileDepthNumber
    # Oversampling: Quick and dirty method = Select N random samples matching size of smallest class
    # Use Tiger's script for creating new fileDepthNumber field; Dorin's new stuff too?

    # Extract relevant feature columns i.e. numerical fields
    cols = ["fixLineNum", "fixNodeLength", "fixNodeStartChar", "bugNodeLength", "bugNodeStartChar", "bugLineNum",
            "fileDepthNumber"]
    X = df[cols]  # X = the feature set
    y = df["bugType"]  # y = target variable (bugType)

    # TODO: Implement feature selection method for creating control model. Else: pre-select based on analysis
    # Use feature selection to choose from 'cols' features which would give good accuracy to a model
    # Preliminary manual analysis suggests ["fixNodeLength", "bugNodeLength", "bugNodeStartChar", "bugLineNum"]

    # Split data to 80:20 train:test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model 1: Control model - does not include fileDepthNumber feature
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Model 2: Variation model of control with fileDepthNumber feature included
    # TODO: Make new X set which is equal to X from Model 1 + fileDepthNumber column
    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)

    # Gather feature importance and model accuracy scores
    importance1 = pd.Series(clf.feature_importances_, index=X.columns)
    importance2 = pd.Series(clf2.feature_importances_, index=X.columns)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc2 = metrics.accuracy_score(y_test, y_pred2)

    # (Debug) Print output to console
    print("(Control) Feature importances \n")
    print(str(importance1))
    print("(Control) Model accuracy: " + str(acc))
    print("(+ fileDepthNumber) Feature importances \n")
    print(str(importance2))
    print("(+ fileDepthNumber) Model accuracy: " + str(acc2))

    # Write results to output file
    with open(output_file, encoding="utf8") as f:
        f.write("(Control) Feature importances \n")
        f.write(str(importance1))
        f.write("(Control) Model accuracy: " + str(acc))
        f.write("(+ fileDepthNumber) Feature importances \n")
        f.write(str(importance2))
        f.write("(+ fileDepthNumber) Model accuracy: " + str(acc2))
        f.close()


def calculate_file_depth(data):
    # Based on Tiger's script: calculate and append fileDepthNumber to data list
    with open(data, encoding="utf8") as f:
        d = json.load(f)

    datacopy = d.copy()

    for x in datacopy:
        filedepth = x["bugFilePath"].count("/")
        x.add("fileDepthNumber", filedepth)

    return datacopy


def main():
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument(
        '--data',
        default='newsstubssmall.json',
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
    # run_experiment(args.data, args.output)


if __name__ == '__main__':
    main()
