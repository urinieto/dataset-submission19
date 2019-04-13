"""
Created 04-06-19 by Matt C. McCallum
"""


# Local imports
from harmonix_dataset import HarmonixDataset

# Third party imports
import librosa
import mir_eval
import matplotlib.pyplot as plt
import madmom
import numpy as np

# Python standard library imports
import argparse
import os
import logging
import multiprocessing
import copy
import tempfile
import pickle
from datetime import datetime


LIBROSA_ALG = 'librosa'
MADMOM_ALG_1 = 'madmom_1'
MADMOM_ALG_2 = 'madmom_2'
MADMOM_ALG_3 = 'madmom_3'
MADMOM_ALG_4 = 'madmom_4'
ESTIMATE_SET_TYPE = {
        LIBROSA_ALG: [],
        MADMOM_ALG_1: [],
        MADMOM_ALG_2: [],
        MADMOM_ALG_3: [],
        MADMOM_ALG_4: []
    }

logging.basicConfig(level=logging.INFO)


def process_algorithms(id_name, audio_dir):
    """
    """
    estimates = copy.deepcopy(ESTIMATE_SET_TYPE)

    logging.info('Analyzing track: {}'.format(id_name))

    # Get audio filename
    audio_fname = os.path.join(audio_dir, id_name + '.mp3')

    # Load audio
    audio_samples = librosa.load(audio_fname)

    # librosa
    _, estimates[LIBROSA_ALG] = librosa.beat.beat_track(audio_samples, units='time')

    return id_name, estimates


def main(audio_dir=None,
         results_dir=None):
    """
    """
    # For each track, load in audio, reference beat vectors, and get estimated beat vectors
    dataset = HarmonixDataset()
    reference_data = dataset.beat_time_lists
    reference_data = {os.path.splitext(os.path.basename(fname))[0]: value for fname, value in reference_data.items()}

    # NOTE [matt.c.mccallum 04.06.19]: The below doesn't seem to work with librosa audio reading, some issue with deadlocks and the librosa load function I think.
    args = [(id_name, audio_dir) for id_name in reference_data.keys()]
    the_pool = multiprocessing.pool.Pool(12, maxtasksperchild=1)
    print(len(args))
    estimates = the_pool.starmap(process_algorithms, args)
    the_pool.close()
    the_pool.join()

    # Save estimates to file
    now = datetime.now()
    date_marker_string = now.strftime("%Y_%m_%d_%H_%M_%S_")
    results_file = os.path.join(results_dir, date_marker_string + "Beat_Tracking_Estimates.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(estimates, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Add to each evaluation metric
    F_MEASURE = 'F-Measure'
    CEMGILS = "Max Cemgil's Score"
    GOTOS = "Goto's Score"
    PSCORE = "McKinney's P-score"
    CONTINUITY = 'Max Continuity-based Score'
    INFORMATION_GAIN = 'Information Gain'
    results = {
        F_MEASURE: copy.deepcopy(ESTIMATE_SET_TYPE),
        CEMGILS: copy.deepcopy(ESTIMATE_SET_TYPE),
        GOTOS: copy.deepcopy(ESTIMATE_SET_TYPE),
        PSCORE: copy.deepcopy(ESTIMATE_SET_TYPE),
        CONTINUITY: copy.deepcopy(ESTIMATE_SET_TYPE),
        INFORMATION_GAIN: copy.deepcopy(ESTIMATE_SET_TYPE)
    }
    for id_name, estimate_set in estimates:
        reference_beats = reference_data[id_name]
        for algorithm, estimated_beats in estimate_set.items():

            mir_eval.beat.validate(reference_beats, estimated_beats)

            results[F_MEASURE][algorithm] += [mir_eval.beat.f_measure(reference_beats, estimated_beats)]

            _, x = mir_eval.beat.cemgil(reference_beats, estimated_beats)
            results[CEMGILS][algorithm] += [x]

            results[GOTOS][algorithm] += [mir_eval.beat.goto(reference_beats, estimated_beats)]

            results[PSCORE][algorithm] += [mir_eval.beat.p_score(reference_beats, estimated_beats)]

            _, _, x, _ = mir_eval.beat.continuity(reference_beats, estimated_beats)
            results[CONTINUITY][algorithm] += [x]

            results[INFORMATION_GAIN][algorithm] += [mir_eval.beat.information_gain(reference_beats, estimated_beats)]

    # Save results struct to file
    results_file = os.path.join(results_dir, date_marker_string + "Beat_Tracking_Results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Estimates beat positions and evaluates beat tracking metrics performance over the harmonix dataset')
    parser.add_argument('--audio-dir', default='../dataset/audio', type=str)
    parser.add_argument('--results-dir', default='../results', type=str)
    kwargs = vars(parser.parse_args())
    main(**kwargs)
