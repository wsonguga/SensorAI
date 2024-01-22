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

# Generate a list of 100 random 0s and 1s.
zero_count = random.randint(0, 65)
one_count = 100 - zero_count

rand_list = [0]*zero_count + [1]*one_count
random.shuffle(rand_list)

print(rand_list)

# Filter a noisy signal.
T = 1
nsamples = 150 #int(T * fs)
t = np.linspace(0, T, nsamples, endpoint=False)

# Create an empty list
data_list = []

for r in rand_list:
    if r == 0:
        sig = np.sin(2*np.pi*10*t) # 10 Hz
        sig = np.append(sig,[0])
        s = list(sig)
    else:
        sig = np.sin(2*np.pi*30*t) # 30 Hz
        sig = np.append(sig,[1])
        s = list(sig)
    data_list.append(s)

data = np.array(data_list)
print("data shape is ",data.shape)
np.save(p / "syn2class",data)