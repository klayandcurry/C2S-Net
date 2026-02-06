import numpy as np
import math
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def snr_calculate(label, data):
    s = np.sum(label ** 2)
    n = np.sum((label - data) * (label - data))
    snr = math.log(s / n, 10)
    snr = snr * 10
    return snr
