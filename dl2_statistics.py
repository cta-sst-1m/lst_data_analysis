#!/usr/bin/env python

"""
Usage:
  ./dl2_data_analysis.py --date=YYYMMDD --model=MODEL
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel
from scipy.optimize import curve_fit
from dateutil.parser import parse as date_parse


def set_figures():
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['legend.numpoints'] = 1 #aby se u errorbaru a bodovych grafu nezobrazoval kazdy bod 2x
    mpl.rcParams['lines.markersize'] = 15
    mpl.rcParams['legend.fontsize'] = 12


def real_data_prepare(filelist, key=None):
    data_merged = None
    for data, i in zip(filelist, range(len(filelist))):
        # nacteni pouze sloupcu s parametry do pandas dataframe
        print(data)
        if '.h5' in data:
            if not os.path.isfile(data):
                print('WARNING: skip file', data, 'as it was not found.')
                continue
            param = pd.read_hdf(data, key=key)

            # zakladni statistika dat
            print('Filename:', data)
            print('Size of dataset:', param.shape[0])

            if data_merged is None:
                data_merged = param
            else:
                data_merged = pd.concat([data_merged, param])
                data_merged = data_merged.reset_index(drop=True)

    print('Size of ALL RUNS MERGED dataset:', data_merged.shape[0])
    return data_merged

def get_run_file_and_length(filename, path):
    filenames = []
    length_s = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() ==  "":
                continue
            else:
                filename, duration_str = line.split()
                filenames.append(path + "/" + filename)
                duration = date_parse(duration_str) - date_parse("0:0:0")
                length_s += duration.total_seconds()
    return filenames, length_s


def histogram_data(data, bins=np.linspace(0, 1, 51)):
    hist = np.histogram(data, bins=bins)[0]
    errors = np.sqrt(hist)
    return hist, errors


def quality_cuts(data,
                 intensity_cut=300,
                 leak_cut=0.1,
                 islands_cut=2,
                 ):
    #mask = (10**data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    mask = (data.intensity >= intensity_cut) & (data.leakage <= leak_cut) & (data.n_islands <= islands_cut)
    return mask


def theta2_cut(data, resolution_file, default_cut=0.35):
    n_event = len(data)
    mask_signal_on = np.zeros(n_event, dtype=bool)
    resol_data = np.loadtxt(resolution_file)
    for ebin_min, ebin_max, resol_68 in resol_data:
        in_current_bin = (data['reco_energy'] >= ebin_min) & (data['reco_energy'] < ebin_max)
        if np.isfinite(resol_68):
            pass_theta2_cut = in_current_bin & (data['theta2'] < resol_68)
        else:
            pass_theta2_cut = in_current_bin & (data['theta2'] < default_cut)
        mask_signal_on[pass_theta2_cut] = True
    return mask_signal_on

        
def gh_cut(data, sensitivity_file, default_cut=0.5):
    n_event = len(data)
    mask_is_gamma = np.zeros(n_event, dtype=bool)
    sensitivity_data = np.loadtxt(sensitivity_file)
    for ebin_min, ebin_max, gh_cut, sens in sensitivity_data:
        in_current_bin = (data['reco_energy'] >= ebin_min) & (data['reco_energy'] < ebin_max)
        if np.isfinite(gh_cut):
            pass_gammaness_cut = in_current_bin & (data['gammaness'] >= gh_cut)
        else:
            pass_gammaness_cut = in_current_bin & (data['gammaness'] >= default_cut)
        mask_is_gamma[pass_gammaness_cut] = True
    return mask_is_gamma


if __name__ == '__main__':
    set_figures()
    
    model = 'jakub'
    all_files_on = []
    all_time_on = 0
    all_files_off = []
    all_time_off = 0
    for date in (20200218, 20200217, 20200215, 20200131, 20200128, 20200127, 20200118, 20200117, 20200115, 20191126, 20191124, 20191123):
    #for date in (20191123,):
        path_dl2 = 'real_data/DL2/' + str(date)
        path_run_info = 'real_data/runs/' + str(date)
        file_on_data_list = path_run_info + '/files_on.txt'
        file_off_data_list = path_run_info + '/files_off.txt'
        path_dl2_data = path_dl2 + '/' + model
        on_data_list, obs_time_on = get_run_file_and_length(file_on_data_list, path_dl2_data)
        off_data_list, obs_time_off = get_run_file_and_length(file_off_data_list, path_dl2_data)
        all_files_on.extend(on_data_list)
        all_time_on += obs_time_on
        all_files_off.extend(off_data_list)
        all_time_off += obs_time_off
    on_data = real_data_prepare(all_files_on, key='dl1/event/telescope/parameters/LST_LSTCam')
    off_data = real_data_prepare(all_files_off, key='dl1/event/telescope/parameters/LST_LSTCam')
    
    source_position = [0, 0]
    f = 0.1 / 0.05  # deg / m
    on_data['theta2'] = f**2 * ((source_position[0]-on_data['reco_src_x'])**2 + (source_position[1]-on_data['reco_src_y'])**2)
    off_data['theta2'] = f**2 * ((source_position[0]-off_data['reco_src_x'])**2 + (source_position[1]-off_data['reco_src_y'])**2)
    on_data['alt_tel_deg'] = on_data['alt_tel'] / np.pi * 180.
    off_data['alt_tel_deg'] = off_data['alt_tel'] / np.pi * 180.

    quality_mask_on = quality_cuts(on_data, intensity_cut=300, leak_cut=0.1, islands_cut=2)
    quality_mask_off = quality_cuts(off_data, intensity_cut=300, leak_cut=0.1, islands_cut=2)

    file_resol = 'models/' + model + '/disp_rf/resolution_68.txt'
    theta2_mask_on = theta2_cut(on_data, file_resol, default_cut=1.0)
    theta2_mask_off = theta2_cut(off_data, file_resol, default_cut=1.0)

    file_gh_cuts = 'models/' + model + '/gh_sep_rf/sensitivity_optimized.dat'
    isgamma_mask_on = gh_cut(on_data, file_gh_cuts, default_cut=0.5)
    isgamma_mask_off = gh_cut(off_data, file_gh_cuts, default_cut=0.5)
    
    alt_bins=np.linspace(0, 90, 91)
    alt_mid = .5 * (alt_bins[:-1] + alt_bins[1:])
    hist_on, _ = np.histogram(on_data['alt_tel_deg'], alt_bins)
    print('ON:', np.sum(hist_on), 'events')
    hist_quality_on, _ = np.histogram(on_data[quality_mask_on]['alt_tel_deg'], alt_bins)
    print('ON, QC:', np.sum(hist_quality_on), 'events ')
    hist_gamma_on, _ = np.histogram(on_data[quality_mask_on & isgamma_mask_on]['alt_tel_deg'], alt_bins)
    print('ON, gamma:', np.sum(hist_gamma_on), 'events ')
    hist_theta2_on, _ = np.histogram(on_data[quality_mask_on & isgamma_mask_on & theta2_mask_on]['alt_tel_deg'], alt_bins)
    print('ON, gamma on source:', np.sum(hist_theta2_on), 'events ')
    hist_off, _ = np.histogram(off_data['alt_tel_deg'], alt_bins)
    print('OFF:', np.sum(hist_off), 'events ')
    hist_quality_off, _ = np.histogram(off_data[quality_mask_off]['alt_tel_deg'], alt_bins)
    print('OFF, QC:', np.sum(hist_quality_off), 'events ')
    hist_gamma_off, _ = np.histogram(off_data[quality_mask_off & isgamma_mask_off]['alt_tel_deg'], alt_bins)
    print('OFF, gamma:', np.sum(hist_gamma_off), 'events ')
    hist_theta2_off, _ = np.histogram(off_data[quality_mask_off & isgamma_mask_off & theta2_mask_off]['alt_tel_deg'], alt_bins)
    print('OFF, gamma on source:', np.sum(hist_theta2_off), 'events ')

    fig = plt.figure()
    #plt.step(alt_mid, hist_on, where='mid', label='ON')
    plt.step(alt_mid, hist_quality_on, where='mid', label='ON, quality cuts')
    plt.step(alt_mid, hist_gamma_on, where='mid', label='ON, $\gamma$ness cuts')
    plt.step(alt_mid, hist_theta2_on, where='mid', label='ON, $\Theta^2$ cuts')
    #plt.step(alt_mid, hist_off, where='mid', label='OFF')
    plt.step(alt_mid, hist_quality_off, where='mid', label='OFF, quality cuts')
    plt.step(alt_mid, hist_gamma_off, where='mid', label='OFF, $\gamma$ness cuts')
    plt.step(alt_mid, hist_theta2_off, where='mid', label='OFF, $\Theta^2$ cuts')
    plt.ylabel('#events')
    plt.xlabel('altitude (deg)')
    plt.xlim([0, 90])
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('altitude_stats.png')
    plt.close(fig)
