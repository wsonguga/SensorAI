import numpy as np
import h5py
from pathlib import Path


p = Path('.')
datapath = p / "AI_engine/test_data/"

filename = "CAP.h5"
data = h5py.File(datapath/filename, 'r') # 'r' is to read
print(type(data))

for key in data.keys():
    print(key)

print(data['Machine0'])
m0 = data['Machine0']
for key in m0:
    print(key)

m0_data = np.array(m0['data'])
m0_labels = np.array(m0['labels'])

print(len(m0_labels)) # 12,769 labels
print(m0_labels.shape) # (12769, 1)
print(m0_data.shape) # (12769, 3000, 19)
print(m0_data[0].shape) # (3000, 19)