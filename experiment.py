"""
Experiment implementation for MSR Mining Challenge 2021
- Compares 2 Random Forest classifiers trained with and without file depth
- Output: Feature importance weighting and model accuracy scores for each model for comparison
"""

import json
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def run_experiment(data):
    """Run the Random Forest model experiment."""

    # Load data from JSON file
    df = pd.read_json(data)
    print(df.head())  # For debug/ref: print first 5 data entries from JSON to check format is valid

    # Extract relevant feature columns; TODO - implement method/code to exclude feature(s) of choice
    cols = ["fixLineNum", "fixNodeLength", "fixNodeStartChar", "bugNodeLength", "bugNodeStartChar", "bugLineNum",
            "fileDepthNumber"]
    X = df[cols]
    y = df["bugType"]

    # Split data to 80:20 train:test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Gather feature importance and model accuracy scores
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    acc = metrics.accuracy_score(y_test, y_pred)

    # TODO: Output scores nicely e.g. print to console formatted or write to file


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument(
        '--data',
        default='newsstubssmall.json',
        type=str,
        help='Path to data (JSON)'
    )
    parser.add_argument(
        '--exclude_feature',
        default='fileDepthNumber',
        type=str,
        help='Name of feature(s) to exclude from model'
    )
    args = parser.parse_args()

    # TODO: Run experiment function on input parameters
    # run_experiment(args.data)


if __name__ == '__main__':
    main()
