# -*- coding: UTF-8 -*-
#!/usr/bin/python

import sys
import XGBoost
import GetFeatures


def get_sepsis_score(input_file):
    data= GetFeatures.readData(input_file)
    if data[0].size!=0:
        feature = GetFeatures.getFeature(data[0][:, :-1])
        # generate predictions
        (scores, labels) = XGBoost.predict_exist_XGBoost(feature,0.1)
    return (scores, labels)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('make sure: python get_sepsis_score.py input.psv output.psv')
    input_file = sys.argv[1]
    (scores, labels) = get_sepsis_score(input_file)

    # write predictions to output file
    output_file = sys.argv[2]
    with open(output_file, 'w') as f:
        if labels.size!=0:
            f.write('PredictedProbability|PredictedLabel\n')
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))
