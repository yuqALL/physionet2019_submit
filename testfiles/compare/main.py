#!/usr/bin/env python

import numpy as np, os, sys
from get_sepsis_score import genFeature, imputer_missing_mean, imputer_missing_median
from GetFeatures import getFeature


def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data


if __name__ == '__main__':
    # Parse arguments.

    input_directory = "./data/"

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'psv'):
            files.append(f)

    # # Iterate over files.
    # for f in files:
    #     # Load data.
    #     input_file = os.path.join(input_directory, f)
    #     data = load_challenge_data(input_file)
    #
    #     # Make predictions.
    #     num_rows = len(data)
    #     scores = np.zeros(num_rows)
    #     labels = np.zeros(num_rows)
    #     feature1 = getFeature(data, False)
    #     feature1[:, 0:34] = imputer_missing_mean(feature1[:, 0:34])
    #     feature1[:, 34:] = imputer_missing_median(feature1[:, 34:])
    #     feature2 = np.zeros_like(feature1)
    #     for t in range(num_rows):
    #         current_data = data[:t + 1]
    #         feature2[:t + 1] = genFeature(current_data)
    #         feature2[:t + 1, 0:34] = imputer_missing_mean(feature2[:t + 1, 0:34])
    #         feature2[:t + 1, 34:] = imputer_missing_median(feature2[:t + 1, 34:])
    #
    #     result = feature2 - feature1

    # Load data.
    input_file = os.path.join(input_directory, files[0])
    data = load_challenge_data(input_file)

    # Make predictions.
    num_rows = len(data)
    scores = np.zeros(num_rows)
    labels = np.zeros(num_rows)
    feature1 = getFeature(data, False)
    feature1[:, 0:34] = imputer_missing_mean(feature1[:, 0:34])
    feature1[:, 34:] = imputer_missing_median(feature1[:, 34:])
    feature2 = np.zeros_like(feature1)
    for t in range(num_rows):
        current_data = data[:t + 1]
        feature2[t, :] = genFeature(current_data)
        feature2[:t + 1, 0:34] = imputer_missing_mean(feature2[:t + 1, 0:34])
        feature2[:t + 1, 34:] = imputer_missing_median(feature2[:t + 1, 34:])

    result = feature2 - feature1
