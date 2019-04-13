"""
Created 04-06-19 by Matt C. McCallum
"""


# Local imports
# None.

# Third party imports
import pandas as pd
import numpy as np

# Python standard library imports
import os

class HarmonixDataset(object):
    """
    An object for interfacing with the dataset data.
    """

    def __init__(self, dataset_dir='../dataset'):
        """
        Constructor.

        Args:
        
        """
        # Define dataset info
        self._DATA_DIR = os.path.abspath(dataset_dir)
        self._BEAT_DIR = os.path.join(self._DATA_DIR, 'beats')
        self._BEAT_MARKER_COLUMN = 'BeatMarker'
        self._BEAT_NUMBER_COLUMN = 'BeatNumber'
        self._BAR_NUMBER_COLUMN = 'BarNumber'
        self._BEATS_COLUMNS = [self._BEAT_MARKER_COLUMN, self._BEAT_NUMBER_COLUMN, self._BAR_NUMBER_COLUMN]
        self._SEGMENT_DIR = os.path.join(self._DATA_DIR, 'segments')
        self._SEG_BOUNDARY_COLUMN = 'SegmentStart'
        self._SEG_LABEL_COLUMN = 'SegmentLabel'
        self._SEGMENTS_COLUMNS = [self._SEG_BOUNDARY_COLUMN, self._SEG_LABEL_COLUMN]

        # Load entire dataset into memory
        self._beat_files = [os.path.join(self._BEAT_DIR, fname) for fname in os.listdir(self._BEAT_DIR)]
        self._seg_files = [os.path.join(self._SEGMENT_DIR, fname) for fname in os.listdir(self._SEGMENT_DIR)]
        self._beat_data = {os.path.basename(fname):pd.read_csv(fname, names=self._BEATS_COLUMNS, delimiter='\t') for fname in self._beat_files}
        self._seg_data = {os.path.basename(fname):pd.read_csv(fname, names=self._SEGMENTS_COLUMNS, delimiter=' ') for fname in self._seg_files}

    @property
    def beat_dataframe(self):
        """
        """
        return self._beat_data

    @property
    def segment_dataframe(self):
        """
        """
        return self._seg_data

    @property
    def beat_time_lists(self):
        """
        """
        return {fname: data[self._BEAT_MARKER_COLUMN].values for fname, data in self._beat_data.items()}

    def downbeat_time_lists(self, offset):
        """
        """
        downbeats_each_track = {}
        for fname, df in self._beat_data.items():
            bar_numbers = np.array(df[self._BAR_NUMBER_COLUMN])
            bar_start_idxs = np.argwhere((bar_numbers[1:]-bar_numbers[:-1])>0) + offset # <= We ignore the last bar as it is usually incomplete - e.g., the final beat
            if bar_numbers[0] == 1:
                bar_start_idxs = np.concatenate((np.array([0]), bar_start_idxs.flatten()))
            downbeats_each_track[fname] = df[self._BEAT_MARKER_COLUMN].values[bar_start_idxs].flatten()
        return downbeats_each_track
