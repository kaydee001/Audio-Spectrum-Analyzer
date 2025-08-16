"""
audio spectrum analyzer - loads .wav file and visualizes time + frequency domains
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# input audio file 
audio_file = input("Enter audio file path: ")
sample_rate, data = wavfile.read(audio_file)

try:
    audio_file = input("Enter audio file path: ")
    sample_rate, data = wavfile.read(audio_file)
except FileNotFoundError:
    print("Audio file not found!")
    exit()

# since audio shape is 2 (dual channel or STEREO)
left_channel = data[:, 0]

# time domain
time = np.arange(len(left_channel)) / sample_rate

# normalization (i.e. we need to keep the values b/w -1 to +1)
left_channel = left_channel.astype(float) / np.iinfo(np.int16).max

# change from time domain to freq domain
fft_res = np.fft.fft(left_channel)
magnitude_spectrum = np.abs(fft_res)
freqs = np.fft.fftfreq(len(fft_res), d=1/sample_rate)

# to only plot the positive frequencies
nyquist_samples = len(magnitude_spectrum)//2

# subplot the plots as 2 rows, 1 column 
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

fig.suptitle('audio spectrum analyzer')

# plotting audio waveform 
axs[0].plot(time, left_channel,color="#00fff2")
axs[0].set_xlabel("time (s)")
axs[0].set_ylabel("amplitude")
axs[0].grid(alpha=0.3)
axs[0].set_title("audio waveform")

# plotting frequency spectrum 
axs[1].plot(freqs[:nyquist_samples], magnitude_spectrum[:nyquist_samples], color="#ff00ea")
axs[1].set_xlabel("frequency (hz)")
axs[1].set_ylabel("magnitude")
axs[1].grid(alpha=0.3)
axs[1].set_title("frequency spectrum")

fig.tight_layout(pad=0.5)
plt.show()