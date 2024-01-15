import os
import numpy as np

def load_scg(noise_level, train_or_test: str):
    """
    Load SCG (SeismoCardioGram) data with specified noise level and training/testing mode.

    Args:
        noise_level (float): The level of noise in the data (0, 0.1, or 0.8).
        train_or_test (str): Either 'train' or 'test' mode.

    Returns:
        signals (numpy.ndarray): Loaded SCG signals.
        labels (numpy.ndarray): Loaded labels (ID + Time + H + R + S + D).
        duration (int): Duration of the data in seconds (10 s).
        fs (int): Sampling frequency of the data (100 Hz).
    """
    # Check if the provided train_or_test is valid
    if train_or_test.lower() not in ['train', 'test']:
        raise ValueError("Please make sure it is either 'train' or 'test'!")

    # Check if the provided noise_level is valid
    if noise_level not in [0, 0.1, 0.8]:
        raise ValueError("Now, we only support noise levels 0, 0.1, and 0.8")

    n_samples, S_start, S_end = 0, 0, 0

    # Set the number of samples and range based on train_or_test mode
    if train_or_test.lower() == 'train':
        n_samples = 5000
        S_start, S_end = 90, 140
    elif train_or_test.lower() == 'test':
        n_samples = 3000
        S_start, S_end = 141, 178

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    filepath = 'sim_{}_{}_{}_{}_{}.npy'.format(n_samples, noise_level, S_start, S_end, train_or_test)

    # Combine the current file's directory and the constructed file path
    file_path = os.path.join(os.path.dirname(current_file_path), 'data', filepath)

    # Load data from the constructed file path
    data = np.load(file_path)

    # Split loaded data into signals and labels
    signals, labels = data[:, :1000], data[:, 1000:]

    # Set duration and sampling frequency
    duration = 10
    fs = 100

    return signals, labels, duration, fs

def load_scg_template(noise_level, train_or_test: str):

    if train_or_test.lower() not in ['train', 'test']:
        raise ValueError("Please make sure it is either 'train' or 'test'!")

    # Check if the provided noise_level is valid
    if noise_level not in [0.1]:
        raise ValueError("Now, we only support noise levels 0.1")

    n_samples, S_start, S_end = 0, 0, 0
    # Set the number of samples and range based on train_or_test mode

    S_start, S_end = 90, 180
    if train_or_test.lower() == 'train':
        n_samples = 5000
    elif train_or_test.lower() == 'test':
        n_samples = 3000
    filepath = 'sim_{}_0.1_{}_{}_{}_template.npy'.format(n_samples, S_start, S_end, train_or_test)

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Combine the current file's directory and the constructed file path
    file_path = os.path.join(os.path.dirname(current_file_path), 'classification_S', filepath)

    # Load data from the constructed file path
    data = np.load(file_path, allow_pickle=True)

    signals = []
    labels = []

    for piece in data:
        piece_np = np.array(piece)
        signals.append(piece_np[:-6])
        labels.append(piece_np[-6:])

    # Set duration and sampling frequency
    duration = 10
    fs = 100

    return signals, labels, duration, fs

if __name__ == '__main__':
    load_scg(0.1, 'test')
    load_scg_template(0.1, 'test')
