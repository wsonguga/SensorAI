# Package Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum
import random

from scipy import signal
from scipy.signal import butter, lfilter, cheby1, cheby2
from numpy import array

p = Path('.')

datapath = p / "AI_engine/test_data/"

data = np.load(datapath/'synthetic_dataset.npy')

count = 0
for d in data:
    if count == 0:
        print("d shape is ",d.shape)
        print("last d value is ",d[-1])
        print("data 0 shape is ",data[count,:].shape)
        print("data 0 label is ",data[count,-1])
    if d[-1] != 0:
        data[count,-1] = 1
    count += 1

np.save(datapath/'qi2class.npy', data)
