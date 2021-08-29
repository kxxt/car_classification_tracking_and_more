from scipy.optimize import curve_fit
import numpy as np
import json


def find_fit(path):
    '''
    Return the best fit of A, B, C and D
    :param path: The file path of training data json
    '''

    def transform(x, a, b, c, d):
        return a * np.arctan(b * x + c) + d

    with open(path) as training_file:
        pre_data = training_file.read()

    pre_data = json.loads(pre_data)

    # pick out x and y as two lists:
    xdata, ydata = zip(*pre_data)

    # xdata = list(xdata)
    # ydata = list(ydata)

    output = curve_fit(transform, xdata, ydata)[0]
    return list(output)

if __name__ == "__main__" :
    
    PATH = "velocity_detection/training_data.json"

    output = find_fit(PATH)
    print(output)

