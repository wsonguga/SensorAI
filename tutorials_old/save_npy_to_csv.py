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

data = np.loadtxt(datapath/'syn_data.csv',delimiter=',')

np.save(datapath/'syn_data.npy', data)

#datapath / "synthetic_dataset.npy"