"""
Experiment implementation for MSR Mining Challenge 2021
- Compares 2 Random Forest classifiers trained with and without file depth
- Output: Feature importance weighting and model accuracy scores for each model for comparison
"""

import json
import argparse

from sklearn.ensemble import RandomForestClassifier

def run_experiment(data, exclude_feature):
    """Run the Random Forest model experiment."""
    # TODO
    # - Take in data and name of feature to exclude; preprocess and run for trees
    # - Return feature importance and model accuracy

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

    # Run experiment function on input parameters
    # TODO

if __name__ == '__main__':
    main()
