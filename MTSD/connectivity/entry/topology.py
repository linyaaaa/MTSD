"""
.. _tut-sensor-locations:

Working with sensor locations
=============================

This tutorial describes how to read and plot sensor locations, and how
MNE-Python handles physical locations of sensors.

As usual we'll start by importing the modules we need and loading some
:ref:`example data <sample-dataset>`:
"""

# %%

import os
import numpy as np
import matplotlib.pyplot as plt
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

montage_dir = os.path.join(os.path.dirname(mne.__file__),
                           'channels', 'data', 'montages')
print('\nBUILT-IN MONTAGE FILES')
print('======================')
print(sorted(os.listdir(montage_dir)))

biosemi_montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
print(biosemi_montage)
fig, returns = biosemi_montage.plot(kind='topomap', show_names=True)
plt.show()
print(returns)
