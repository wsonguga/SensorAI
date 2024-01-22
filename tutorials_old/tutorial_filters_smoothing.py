import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum

from scipy import signal
from scipy.signal import butter, lfilter, cheby1, cheby2
from numpy import array

from sklearn.model_selection import train_test_split
import filter_builder as fltr
import tutorials_old.load_data as ld

# Wiener
def testWiener(y):
    wi = signal.wiener(y, mysize=11)
    return wi

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/synthetic_dataset.npy")
#data = np.load(datapath / "synthetic_dataset.npy")
data = ld.selectFileAndLoad()


#print("shape of  data is ",data.shape)

x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # label

#print("shape of x is ",x.shape)
#print("shape of y is ",y.shape)

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Use line below with PV_Data Only
#x = np.delete(x, 799999, 1)  # delete second column of C

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Filter a noisy signal
# T = 1
# nsamples = 100 #int(T * fs)
# t = np.linspace(0, T, nsamples, endpoint=False)
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t) # 10 Hz + 20 Hz + 30 Hz

sig = X_train[0]
t = np.linspace(0, 1, len(sig), endpoint=False)

plt.figure(1)
plt.clf()
plt.plot(t, sig, label='Noisy signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()

wi = fltr.gaussian_spline(sig,2)

print(sig.shape)
print(type(wi))
print(wi.shape)

plt.figure(2)
plt.clf()
plt.plot(t, wi, label='Wiener filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()