from scipy.optimize import curve_fit
import numpy as np
import json


def find_fit(path_or_data):
    """
    Return the best fit of A, B, C and D
    :param path_or_data: The file path of training data json or the data
    """

    def transform(x, a, b, c, d):
        return a * np.arctan(b * x + c) + d

    if isinstance(path_or_data, str):
        with open(path_or_data) as training_file:
            pre_data = json.load(training_file)
    else:
        pre_data = path_or_data

    # pick out x and y as two lists:
    xdata, ydata = zip(*pre_data)

    # xdata = list(xdata)
    # ydata = list(ydata)

    output = curve_fit(transform, xdata, ydata)[0]
    return tuple(output)


if __name__ == "__main__":
    PATH = "velocity_detection/training_data.json"

    output = find_fit(PATH)
    print(output)
