import sys
import numpy as np
import pandas as pd
from pandas import read_csv
from pathlib import Path

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

p = Path('.')
datapath = p / "AI_engine/test_data/"

def load(data_filename):
  data_file = np.load(data_filename)
  return data_file

#"""
def selectFileAndLoad():
  Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
  filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
  file = load(filename)
  return file
#"""