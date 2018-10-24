#!/usr/bin/env python3

import math

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

pull_requests_dataframe = pd.read_csv("/pull_requests.csv")

pull_requests_dataframe = pull_requests_dataframe.reindex(
        np.random.permutation(pull_requests_dataframe.index))
pull_requests_dataframe

print(pull_requests_dataframe.describe())
