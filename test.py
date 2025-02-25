import numpy as np
from pyemd import emd

# Define two histograms as numpy arrays
histogram_1 = np.array([0.0, 1.0, 2.0, 0.0])
histogram_2 = np.array([0.0, 0.0, 1.0, 3.0])

# Define the distance matrix (cost to move between bins)
distance_matrix = np.array([
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0],
    [3.0, 2.0, 1.0, 0.0]
])

# Calculate the EMD
emd_value = emd(histogram_1, histogram_2, distance_matrix)

# Print the result
print(f"The Earth Mover's Distance is: {emd_value}")