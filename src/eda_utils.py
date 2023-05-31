"""
This is the EDA module.

This module contains functions that help with plotting audio data
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
from typing import List, Optional, Tuple


def show_sampling(length: float, 
                  sampling_rate: float, 
                  frequency: float, 
                  show_signal: bool = True, 
                  show_sampling: bool = True, 
                  plot_sampling: bool = False) -> None:
    """
    Plot the original signal and the effect of sampling.

    Args:
        length (float): The duration of the signal in seconds.
        sampling_rate (float): The number of samples taken per second.
        frequency (float): The frequency of the original signal in Hz.
        show_signal (bool, optional): Whether to show the original signal. Default is True.
        show_sampling (bool, optional): Whether to show the sampled points. Default is True.
        plot_sampling (bool, optional): Whether to plot the signal after sampling. Default is False.

    Returns:
        None
    """
    
    x = np.linspace(0, length, 10000)
    y = np.sin(2 * np.pi * frequency * x)
    number_of_points = int(length * sampling_rate)
    time = np.linspace(0, length, number_of_points)
    signal = np.sin(2 * np.pi * frequency * time)

    plt.figure(figsize=(20, 6))
    if show_signal:
        plt.plot(x, y, lw=3, label='Original signal')
    if show_sampling:
        plt.scatter(time, signal, color='red', label='Sampling')
    if plot_sampling:
        plt.plot(time, signal, color='purple', label='Signal after sampling')
    plt.legend(loc='upper right')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()

    
def signal_generator(length: float, 
                     sampling_rate: int, 
                     frequencies: List[float], 
                     show_signals: bool = True, 
                     show_signals_sum: bool = False, 
                     split_plots: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a signal consisting of sinusoids with specified frequencies.

    Args:
        length (float): length of the signal in seconds
        sampling_rate (int): sampling rate in Hz
        frequencies (list): list of frequencies in Hz
        show_signals (bool, optional): whether to plot individual signals (default True)
        show_signals_sum (bool, optional): whether to plot the sum of signals (default False)
        split_plots (bool, optional): whether to plot individual signals in separate plots (default False)

    Raises:
        AssertionError: if frequencies is not passed as a list
        
    Returns:
        signal (ndarray): generated signal as a 2D numpy array
        time (ndarray): time axis of the generated signal as a 1D numpy array
    """

    assert isinstance(frequencies, list), 'Frequencies must be passed as a list'
    number_of_points = int(length * sampling_rate)
    time = np.linspace(0, length, number_of_points)
    signal = np.zeros((len(frequencies),number_of_points))
    
    for i, hz in enumerate(frequencies):
        signal[i] += np.sin(2*np.pi*hz*time)
        
    if split_plots and show_signals:
        if show_signals_sum:
            plt.figure(figsize = (20,6))
            plt.plot(time, signal.sum(axis = 0), lw = 3, color = 'red', label = 'sum')
            plt.legend(loc = 'upper right')
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.show()
        for i,sig in enumerate(signal):
                plt.figure(figsize = (20,6))
                plt.plot(time, sig, label = str(frequencies[i]) + ' Hz')
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend(loc = 'upper right')
                plt.show()
    else:
        plt.figure(figsize = (20,6))
        if show_signals:
            for i,sig in enumerate(signal):
                plt.plot(time, sig, label = str(frequencies[i]) + ' Hz')
        if show_signals_sum:
            plt.plot(time, signal.sum(axis = 0), lw = 3, color = 'red', label = 'sum')
        plt.legend(loc = 'upper right')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
        
    return signal, time


def plot_random_spec(df: pd.DataFrame, labels: List[int] = [0, 1]) -> None:
    """
    Plots a random spectrogram for each label in the provided dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset with columns 'path', 'label', and 'species'.
        labels (List[int], optional): The list of labels for which to plot the spectrograms. Default is [0, 1].

    Raises:
        ValueError: If labels contain values that are not present in the 'label' column of the dataframe.

    Returns:
        None
    """
    valid_labels = df['label'].unique()
    if not set(labels).issubset(valid_labels):
        raise ValueError("labels must be a subset of %r." % valid_labels)

    paths = []
    species = []
    for label in labels:
        sample = df[df['label'] == label].sample(1)
        paths.append(sample['path'].values[0])
        species.append(sample['species'].values[0])

    num_paths = len(paths)
    rows = math.ceil(num_paths / 2)
    cols = min(num_paths, 2)
    figure, axes = plt.subplots(rows, cols, figsize=(20, rows * 10 / cols), squeeze=False)
    figure.tight_layout()

    for ax, path, spec in zip(axes.flat, paths, species):
        waveform, sample_rate = torchaudio.load(path)
        spectrum, freqs, t, im = ax.specgram(waveform[0], Fs=sample_rate)
        ax.set_title(spec)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im).set_label('Intensity [dB]')
        
        
    if len(labels)>1 & len(labels)%2!=0:
        figure.delaxes(axes.flat[-1])


    plt.tight_layout()
    plt.show()
    
def plot_spec(files: List[str]) -> None:
    """
    Plot spectrograms of audio files.

    Args:
        files (List[str]): A list of file paths to audio files.

    Raises:
        AssertionError: If files is not a list.

    Returns:
        None
    """
    assert isinstance(files, list), 'Files must be passed as a list of paths'

    num_files = len(files)
    rows = math.ceil(num_files / 2)
    cols = min(num_files, 2)
    figure, axes = plt.subplots(rows, cols, figsize=(20, rows * 10 / cols), squeeze=False)
    figure.tight_layout()

    for ax, file in zip(axes.flat, files):
        waveform, sample_rate = torchaudio.load(file)
        spectrum, freqs, t, im = ax.specgram(waveform[0], Fs=sample_rate)
        ax.set_title(file.split('/')[-1])
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.colorbar(im).set_label('Intensity [dB]')
    
    if num_files>1 & num_files%2!=0:
        figure.delaxes(axes.flat[-1])

    plt.tight_layout()
    plt.show()

def plot_waveform(path: str, time: Optional[float] = None) -> None:
    """
    Loads and plots the waveform from an audio file.

    Args:
        path (str): The path of the audio file.
        time (float, optional): The length of the audio (in seconds) to plot.
                               If provided, only the specified duration of the audio will be plotted.
                               If None, the entire audio waveform will be plotted. Default is None.

    Returns:
        None
    """
    waveform, sample_rate = torchaudio.load(path)
    if time is not None:
        num_frames = math.ceil(time * sample_rate)
        waveform = waveform[:, :num_frames]
    
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(20, 6))
    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
