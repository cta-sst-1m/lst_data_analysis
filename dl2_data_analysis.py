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

parser = argparse.ArgumentParser()

parser.add_argument('--date', action='store', type=str,
                    dest='date',
                    default=None
                    )

parser.add_argument('--models', action='store', type=str,
                    dest='models',
                    default=None
                    )

args = parser.parse_args()

def set_figures():
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['legend.numpoints'] = 1 #aby se u errorbaru a bodovych grafu nezobrazoval kazdy bod 2x
    mpl.rcParams['lines.markersize'] = 15
    mpl.rcParams['legend.fontsize'] = 12


def real_data_prepare(filelist, key):
    # merge data from DL2 files in filelist
    data_merged = None
    for i, data in enumerate(filelist):
        print(data)
        if '.h5' in data:
            if not os.path.isfile(data):
                print('WARNING: skip file', data, 'as it was not found.')
                continue
            param = pd.read_hdf(data, key=key)
            print('Filename:', data)
            print('Size of dataset:', param.shape[0])

            if data_merged is None:
                data_merged = param
            else:
                data_merged = pd.concat([data_merged, param])
                data_merged = data_merged.reset_index(drop=True)

    print('Size of ALL RUNS MERGED dataset:', data_merged.shape[0])
    return data_merged


def alpha_angle(data, source_x, source_y): 
    #etienne's code from scan_crab_cluster.c
    d_x = np.cos(data['psi'])
    d_y = np.sin(data['psi'])
    to_c_x = source_x - data['x']
    to_c_y = source_y - data['y']
    to_c_norm = np.sqrt(to_c_x**2.0 + to_c_y**2.0)
    to_c_x = to_c_x/to_c_norm
    to_c_y = to_c_y/to_c_norm
    p_scal_1 = d_x*to_c_x + d_y*to_c_y
    p_scal_2 = -d_x*to_c_x + -d_y*to_c_y
    alpha_c_1 = abs(np.arccos(p_scal_1))
    alpha_c_2 = abs(np.arccos(p_scal_2))
    alpha_cetienne = alpha_c_1
    alpha_cetienne[alpha_c_2 < alpha_c_1] = alpha_c_2[alpha_c_2 < alpha_c_1]
    data['r'] = to_c_norm
    data['miss'] = data['r'] * np.sin(alpha_cetienne)
    data['alpha'] = 180.0/np.pi*alpha_cetienne
    return data


def sigma_lima(N_on, N_off, alpha):
    #significance using Li & Ma
    sigma_lima = np.sqrt(2 * (N_on * np.log((1+alpha)/alpha * N_on / (N_on + N_off)) + N_off * np.log((1+alpha) * (N_off / (N_on + N_off))) ))
    return sigma_lima



def generate_image(data, bins=100, range_axis=np.array([[-1, 1], [-1, 1]]), onoff_factor=1, resolution=0.2):
    # convolving image with gaussian kernel - it effectively handles each event like 2d gaussian instead of a point (sigma = resolution)
    range_x_deg = range_axis[0, 1] - range_axis[0, 0]
    n_pix_deg = bins / range_x_deg  # px / deg
    sigma_px = sigma * n_pix_deg    # px
    kernel_x_size = int(10*sigma_px)
    if kernel_x_size % 2 == 0:  # kernel size must be odd
         kernel_x_size =  kernel_x_size + 1
    kernel_y_size = kernel_x_size
    kernel_x_size_deg = kernel_x_size / n_pix_deg

    kernel = Gaussian2DKernel(sigma_px, mode='oversample', factor=100, x_size=kernel_x_size, y_size=kernel_y_size).array
    heatmap, xedges, yedges = np.histogram2d(data['reco_src_x_deg'], data['reco_src_y_deg'], bins=bins, range=range_axis)
    image = convolve(onoff_factor * heatmap, kernel)
    return image


def function_gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def sigma_map(on_data, off_data, bins=100, range_axis=np.array([[-1, 1], [-1, 1]]), onoff_factor=1, resolution=0.2):
    sigma = resolution
    #bins = 20
    # sigma pro kernel musi byt zadana v pixelech, ne ve stupnich!
    range_x_deg = range_axis[0, 1] - range_axis[0, 0]
    n_pix_deg = bins / range_x_deg  # px / deg
    sigma_px = sigma * n_pix_deg    # px
    kernel_x_size = int(10*sigma_px)
    if kernel_x_size % 2 == 0:  # kernel size must be odd
         kernel_x_size =  kernel_x_size + 1
    kernel_y_size = kernel_x_size
    kernel_x_size_deg = kernel_x_size / n_pix_deg

    # overeno, ze tenhle kernel ma fakt maximum = 1 (staci proste vynasobit gaussovku tou normalizaci na integral)
    kernel = 2 * np.pi * sigma_px**2 * Gaussian2DKernel(sigma_px, mode='oversample', factor=10, x_size=kernel_x_size, y_size=kernel_y_size).array
    heatmap_on, xedges, yedges = np.histogram2d(on_data['reco_src_x_deg'], on_data['reco_src_y_deg'], bins=bins, range=range_axis)
    heatmap_off, xedges, yedges = np.histogram2d(off_data['reco_src_x_deg'], off_data['reco_src_y_deg'], bins=bins, range=range_axis)
    image_on = convolve(heatmap_on, kernel, normalize_kernel=False)
    image_off = convolve(heatmap_off, kernel, normalize_kernel=False)
    sigma_map = sigma_lima(image_on, image_off, onoff_factor)

    # adding sign to significance
    sign_mask = image_on < image_off*onoff_factor
    sigma_map[sign_mask] = -sigma_map[sign_mask]

    # Distribution of significance
    # - should be gaussian with some tail
    sigma_dist = sigma_map.flatten()
    x_mid = (xedges[1:]+xedges[:-1])/2.0
    y_mid = (yedges[1:]+yedges[:-1])/2.0
    xx_mid, yy_mid = np.meshgrid(x_mid, y_mid)
    xxx_mid = xx_mid.flatten()
    yyy_mid = yy_mid.flatten()
    theta2 = xxx_mid**2 + yyy_mid**2
    return sigma_map


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
    data_masked = data[mask]
    return data_masked


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

    theta_bins = np.linspace(0, 1, 81)
    alpha_bins = np.linspace(0, 90, 91)
    theta_mid = (theta_bins[1:]+theta_bins[:-1])/2.0
    alpha_mid = (alpha_bins[1:]+alpha_bins[:-1])/2.0
    n_image_bins = 100
    path_dl2 = 'real_data/DL2/' + args.date 
    path_run_info = 'real_data/runs/' + args.date

    for model in args.models.split(','):
        file_resol = 'models/' + model + '/disp_rf/resolution_68.txt'
        file_gh_cuts = 'models/' + model + '/gh_sep_rf/sensitivity_optimized.dat'
        file_on_data_list = path_run_info + '/files_on.txt'
        file_off_data_list = path_run_info + '/files_off.txt'
        path_dl2_data = path_dl2 + '/' + model
        on_data_list, obs_time_on = get_run_file_and_length(file_on_data_list, path_dl2_data)
        off_data_list, obs_time_off = get_run_file_and_length(file_off_data_list, path_dl2_data)

        on_data = real_data_prepare(on_data_list, key='dl1/event/telescope/parameters/LST_LSTCam')
        off_data = real_data_prepare(off_data_list, key='dl1/event/telescope/parameters/LST_LSTCam')

        # quality cuts
        on_data = quality_cuts(on_data, intensity_cut=300, leak_cut=0.1, islands_cut=2)
        off_data = quality_cuts(off_data, intensity_cut=300, leak_cut=0.1, islands_cut=2)
        print('minimum intensity:', np.min(on_data['intensity']))

        # gh separation according to optimized cuts
        mask_gammaness_on = gh_cut(on_data, file_gh_cuts)
        mask_gammaness_off = gh_cut(off_data, file_gh_cuts)

        # Fast theta2 computation using focal length (not very precise)
        source_position = [0, 0]
        f = 0.1 / 0.05  # deg / m
        on_data['theta2'] = f**2 * ((source_position[0]-on_data['reco_src_x'])**2 + (source_position[1]-on_data['reco_src_y'])**2)
        off_data['theta2'] = f**2 * ((source_position[0]-off_data['reco_src_x'])**2 + (source_position[1]-off_data['reco_src_y'])**2

        # Alpha angle added to pandas dataframe
        on_data = alpha_angle(on_data, source_x=source_position[0], source_y=source_position[1])
        off_data = alpha_angle(off_data, source_x=source_position[0], source_y=source_position[1])

        # source position calculation
        on_data['reco_src_x_deg'] = f * on_data['reco_src_x']
        on_data['reco_src_y_deg'] = f * on_data['reco_src_y']
        off_data['reco_src_x_deg'] = f * off_data['reco_src_x']
        off_data['reco_src_y_deg'] = f * off_data['reco_src_y']

        # compute normalization of OFF data looking at the normalization region
        theta2_min = 0.25    # 0.7 deg radius
        theta2_max = 0.75
        theta2_mask_on = (on_data['theta2'] > theta2_min) & (on_data['theta2'] < theta2_max)
        theta2_mask_off = (off_data['theta2'] > theta2_min) & (off_data['theta2'] < theta2_max)
        factor = len(on_data[theta2_mask_on]['alpha']) / len(off_data[theta2_mask_off]['alpha'])
        on_sep_thetacut = len(on_data[theta2_mask_on & mask_gammaness_on]['alpha'])
        off_sep_thetacut = len(off_data[theta2_mask_off & mask_gammaness_off]['alpha'])
        if off_sep_thetacut > 0:
            factor_gh =  on_sep_thetacut / off_sep_thetacut
        else:
            factor_gh = np.inf
        print('ON/OFF events ratio in normalization region: {:.2f}'.format(factor))
        print('ON/OFF events ratio in normalization region (after separation): {:.2f}'.format(factor_gh))

        # stuff about analysis on alpha parameter (not used)
        # skalovani alpha histogramu
        alpha_min = 20  # deg
        alpha_max = 70
        alpha_mask_on = (on_data['alpha'] > alpha_min) & (on_data['alpha'] < alpha_max)
        alpha_mask_off = (off_data['alpha'] > alpha_min) & (off_data['alpha'] < alpha_max)
        factor_alpha = len(on_data[alpha_mask_on]['alpha']) / len(off_data[alpha_mask_off]['alpha'])
        on_sep_alphacut = len(on_data[alpha_mask_on & mask_gammaness_on]['alpha'])
        off_sep_alphacut = len(off_data[alpha_mask_off & mask_gammaness_off]['alpha'])
        if off_sep_alphacut > 0:
            factor_alpha_gh = on_sep_alphacut / off_sep_alphacut
        else:
            factor_alpha_gh = np.inf
        factor_bkg_sup = np.inf
        off_rec_h = len(off_data[mask_gammaness_off]['alpha'])
        if off_rec_h > 0:
            factor_bkg_sup = len(off_data['alpha'])/off_rec_h
        print('Background supression with optimized gammaness cut: {:.2f}'.format(factor_bkg_sup))

        # fill histograms according to theta2
        on_thetahist, err_on_thetahist = histogram_data(on_data['theta2'], bins=theta_bins)
        off_thetahist, err_off_thetahist = histogram_data(off_data['theta2'], bins=theta_bins)
        on_thetahist_gh, err_on_thetahist_gh = histogram_data(on_data[mask_gammaness_on]['theta2'], bins=theta_bins)
        off_thetahist_gh, err_off_thetahist_gh = histogram_data(off_data[mask_gammaness_off]['theta2'], bins=theta_bins)

        on_alphahist, err_on_alphahist = histogram_data(on_data['alpha'], bins=alpha_bins)
        off_alphahist, err_off_alphahist = histogram_data(off_data['alpha'], bins=alpha_bins)
        on_alphahist_gh, err_on_alphahist_gh = histogram_data(on_data[mask_gammaness_on]['alpha'], bins=alpha_bins)
        off_alphahist_gh, err_off_alphahist_gh = histogram_data(off_data[mask_gammaness_off]['alpha'], bins=alpha_bins)

        # theta2 cuts
        mask_signal_on_theta = theta2_cut(on_data, file_resol)
        mask_signal_off_theta = theta2_cut(off_data, file_resol)

        # compute significance with theta2
        N_ON_events = len(on_data[mask_gammaness_on & mask_signal_on_theta]['theta2'])
        N_OFF_events = len(off_data[mask_gammaness_off & mask_signal_off_theta]['theta2'])
        alpha = factor_gh
        if N_OFF_events > 0:
            significance = sigma_lima(N_ON_events, N_OFF_events, alpha)
        else:
            significance = np.inf
        print('Detection significance (theta2): {:.3f}'.format(significance))

        # compute significance with alpha
        alpha_signal_cut = 5   # deg
        mask_signal_on_alpha = on_data['alpha'] < alpha_signal_cut
        mask_signal_off_alpha = off_data['alpha'] < alpha_signal_cut
        N_ON_events = len(on_data[mask_gammaness_on & mask_signal_on_alpha]['alpha'])
        N_OFF_events = len(off_data[mask_gammaness_off & mask_signal_off_alpha]['alpha'])
        alpha = factor_alpha_gh
        if N_OFF_events > 0:
            significance = sigma_lima(N_ON_events, N_OFF_events, alpha)
        else:
            significance = np.inf
        print('Detection significance (alpha): {:.3f}'.format(significance))

        # rates
        on_signal_rate = len(on_data[mask_gammaness_on & mask_signal_on_theta])/obs_time_on
        print('ON DL2 rate (additional intensity cut, gh sep, theta2): {:.2f} Hz'.format(on_signal_rate))
        off_signal_rate = factor_gh * len(off_data[mask_gammaness_off & mask_signal_off_theta])/obs_time_on
        print('OFF DL2 rate (additional intensity cut, gh sep, theta2): {:.2f} Hz'.format(off_signal_rate))
        print('Excess rate: {:.3f} Hz, {:.2f} min-1'.format(on_signal_rate - off_signal_rate, (on_signal_rate - off_signal_rate)*60))

        # FIGURES

        # 1D theta2 plots
        fig = plt.figure()
        plt.title('After g/h separation on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        on_rate = on_thetahist_gh / obs_time_on
        err_on_rate = err_on_thetahist_gh / obs_time_on
        plt.step(theta_mid, on_rate, where='mid', color='red', label='ON')
        plt.errorbar(theta_mid, on_rate, yerr=[err_on_rate, err_on_rate], color='red',  ecolor='red', markersize=0, ls='none')
        off_rate = factor_gh * off_thetahist_gh / obs_time_on
        err_off_rate = err_off_thetahist_gh / obs_time_on
        plt.step(theta_mid, off_rate, where='mid', color='black', label='OFF')
        plt.errorbar(theta_mid, off_rate, yerr=[err_off_rate, err_off_rate], color='black',  ecolor='black', markersize=0, ls='none')
        #plt.plot([theta2_signal_cut, theta2_signal_cut], plt.ylim(), '--', label=r'$\Theta^2$ cut')
        plt.ylabel('rate [Hz]')
        plt.xlabel('$\Theta^2$ [deg2]')
        plt.xlim([0, 1])
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/theta2_ghsep-' + args.date + '.png')
        plt.close(fig)

        # 1D alpha plots
        fig = plt.figure()
        plt.title('After g/h separation on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        on_rate = on_alphahist_gh / obs_time_on
        err_on_rate = err_on_alphahist_gh / obs_time_on
        plt.step(alpha_mid, on_rate, where='mid', color='red', label='ON')
        plt.errorbar(alpha_mid, on_rate, yerr=[err_on_rate, err_on_rate], color='red',  ecolor='red', markersize=0, ls='none')
        plt.plot([alpha_signal_cut, alpha_signal_cut], plt.ylim(), '--', label=r'$\alpha$ cut')
        off_rate = factor_gh * off_alphahist_gh / obs_time_on
        err_off_rate = err_off_alphahist_gh / obs_time_on
        plt.step(alpha_mid, off_rate, where='mid', color='black', label='OFF')
        plt.errorbar(alpha_mid, off_rate, yerr=[err_off_rate, err_off_rate], color='black',  ecolor='black', markersize=0, ls='none')
        plt.ylabel('rate [Hz]')
        plt.xlabel('alpha [deg]')
        plt.xlim([0, 90])
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/alpha_ghsep-' + args.date + '.png')
        plt.close(fig)

        # 2D images
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        h, xbin, ybin, _ = plt.hist2d(
            on_data[mask_gammaness_on]['reco_src_x_deg'], 
            on_data[mask_gammaness_on]['reco_src_y_deg'], 
            bins=n_image_bins,
            range=np.array([(-2, 2), (-2, 2)]),
            #norm=mpl.colors.LogNorm(),
        )
        xbin_mid = 0.5 * (xbin[:-1] + xbin[1:])
        ybin_mid = 0.5 * (ybin[:-1] + ybin[1:])
        cbar = plt.colorbar()
        cbar.set_label('N of events')
        plt.title('ON events (gh sep) on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        plt.xlabel('x reco  [deg]')
        plt.ylabel('y reco  [deg]')
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/image_on_ghsep-' + args.date + '.png')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        h, xbin, ybin, _ = plt.hist2d(
            off_data[mask_gammaness_off]['reco_src_x_deg'], 
            off_data[mask_gammaness_off]['reco_src_y_deg'], 
            bins=n_image_bins, 
            range=np.array([(-2, 2), (-2, 2)]),
            #norm=mpl.colors.LogNorm(),
        )
        xbin_mid = 0.5 * (xbin[:-1] + xbin[1:])
        ybin_mid = 0.5 * (ybin[:-1] + ybin[1:])
        cbar = plt.colorbar()
        cbar.set_label('N of events')
        plt.title('OFF events (gh sep) on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        plt.xlabel('x reco  [deg]')
        plt.ylabel('y reco  [deg]')
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/image_off_ghsep-' + args.date + '.png')
        plt.close(fig)

        # 2d smoothed images
        image_range_deg = np.array([[-2, 2], [-2, 2]])
        image_extend = [image_range_deg[0, 0], image_range_deg[0, 1], image_range_deg[1, 1], image_range_deg[1, 0]]
        on_map_convolved = generate_image(on_data[mask_gammaness_on], bins=n_image_bins, range_axis=image_range_deg, onoff_factor=1, resolution=0.15)
        off_map_convolved = generate_image(off_data[mask_gammaness_off], bins=n_image_bins, range_axis=image_range_deg, onoff_factor=factor_gh, resolution=0.15)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow((on_map_convolved - off_map_convolved).T, interpolation='none', extent=image_extend, origin='lower')
        cbar = fig.colorbar(im)
        cbar.set_label('N of events')
        ax.set_xlabel('x reco [deg]')
        ax.set_ylabel('y reco [deg]')
        ax.set_title('Excess events (gh sep) on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/image_ghsep_excess_gauss-' + args.date + '.png')
        plt.close(fig)

        # significance map
        sigma_map = sigma_map(on_data[mask_gammaness_on], off_data[mask_gammaness_off], bins=n_image_bins, range_axis=image_range_deg, onoff_factor=factor_gh, resolution=0.15)

        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(sigma_map.T, interpolation='none', extent=image_extend, origin='lower')
        cbar = fig.colorbar(im)
        cbar.set_label('Significance')
        ax.set_xlabel('x reco [deg]')
        ax.set_ylabel('y reco [deg]')
        ax.set_title('Significance map on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/image_ghsep_sigma-' + args.date + '.png')
        plt.close(fig)

        # spectra
        ebin = np.logspace(-2, 2, 41)
        ebin_mid = 0.5 * (ebin[:-1] + ebin[1:])
        delta_ebin = ebin[1:] - ebin[:-1]

        histo_on, _ = np.histogram(
            on_data[mask_gammaness_on & mask_signal_on_theta]['reco_energy'], 
            ebin
        )
        rate_on = histo_on / obs_time_on
        rate_on_err = np.sqrt(histo_on) / obs_time_on
        histo_off, _ = np.histogram(
            off_data[mask_gammaness_off & mask_signal_off_theta]['reco_energy'], 
            ebin
        )
        rate_off = factor_gh * histo_off / obs_time_on
        rate_off_err = factor_gh * np.sqrt(histo_off) / obs_time_on
        rate_diff = rate_on - rate_off
        rate_diff_err = np.sqrt(rate_off_err**2 + rate_on_err**2)/2
        rate_diff_err[rate_diff < 0] *= 2
        rate_diff[rate_diff < 0] = 0

        #rate function of energy
        fig, ax = plt.subplots(figsize=(6,5))
        #ax.errorbar(ebin_mid, rate_on, xerr=delta_ebin/2, yerr=rate_on_err/2, fmt='.', label='signal, on data ({:.3f} Hz)'.format(np.sum(rate_on)))
        #ax.errorbar(ebin_mid, rate_off, xerr=delta_ebin/2, yerr=rate_off_err/2, fmt='.', label='signal, off data ({:.3f} Hz)'.format(np.sum(rate_off)))
        #ax.errorbar(ebin_mid, rate_diff, xerr=delta_ebin/2, yerr=np.sqrt(rate_off_err**2 + rate_on_err**2)/2, fmt='b', ms=0, ls='none', label='$\Theta^2 < {:.3f} \deg^2~&~\gamma$ness$ > {:.2f}$ ({:.3f} Hz)'.format(theta2_signal_cut, gammaness_cut, np.sum(rate_diff)))
        ax.errorbar(ebin_mid, rate_diff, xerr=delta_ebin/2, yerr=np.sqrt(rate_off_err**2 + rate_on_err**2)/2, fmt='b', ms=0, ls='none', label='$\Theta^2 < r_{68}^2~&~\gamma$ness  optimized ' + '({:.3f} Hz)'.format(np.sum(rate_diff)))
        ax.step(ebin_mid, rate_diff, where='mid', color='b')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('reconstructed energy [TeV]')
        ax.set_ylabel('rate [Hz]')
        ax.set_title('Energy of excess events on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        ax.grid(True)
        ax.set_ylim([1e-4, 1.0])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/spectra-' + args.date + '.png')
        plt.close(fig)

        #spectra with fixed gammaness cuts
        fix_gammaness_cut = 0.38
        mask_gammaness_on_fixed = on_data['gammaness'] >= fix_gammaness_cut
        mask_gammaness_off_fixed = off_data['gammaness'] >= fix_gammaness_cut
        histo_on_fixed, _ = np.histogram(
            on_data[mask_gammaness_on_fixed & mask_signal_on_theta]['reco_energy'], 
            ebin
        )
        rate_on_fixed = histo_on_fixed / obs_time_on
        rate_on_err_fixed = np.sqrt(histo_on_fixed) / obs_time_on
        histo_off_fixed, _ = np.histogram(
            off_data[mask_gammaness_off_fixed & mask_signal_off_theta]['reco_energy'], 
            ebin
        )
        rate_off_fixed = factor_gh * histo_off_fixed / obs_time_on
        rate_off_err_fixed = factor_gh * np.sqrt(histo_off_fixed) / obs_time_on
        rate_diff_fixed = rate_on_fixed - rate_off_fixed
        rate_diff_err_fixed = np.sqrt(rate_off_err_fixed**2 + rate_on_err_fixed**2)/2
        rate_diff_err_fixed[rate_diff_fixed < 0] *= 2
        rate_diff_fixed[rate_diff_fixed < 0] = 0
        fig, ax = plt.subplots(figsize=(6,5))
        ax.errorbar(ebin_mid, rate_diff_fixed, xerr=delta_ebin/2, yerr=np.sqrt(rate_off_err**2 + rate_on_err**2)/2, fmt='b', ms=0, ls='none', label='$\Theta^2 < r_{68}^2~&~\gamma$ness > ' + '{:.2f} ({:.3f} Hz)'.format(fix_gammaness_cut, np.sum(rate_diff_fixed)))
        ax.step(ebin_mid, rate_diff_fixed, where='mid', color='b')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('reconstructed energy [TeV]')
        ax.set_ylabel('rate [Hz]')
        ax.set_title('Energy of excess events on ' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        ax.grid(True)
        ax.set_ylim([1e-4, 1.0])
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_dl2_data + '/spectra-fixed_gamaness-' + args.date + '.png')
        plt.close(fig)

        #center of gravity
        xbin = np.linspace(-1.15, 1.15, n_image_bins)
        ybin = np.linspace(-1.15, 1.15, n_image_bins)
        cog_on, _, _ = np.histogram2d(on_data['x'], on_data['y'], [xbin, ybin])
        cog_off, _, _ = np.histogram2d(off_data['x'], off_data['y'], [xbin, ybin])
        X, Y = np.meshgrid(xbin, ybin)
        fig, axes = plt.subplots(2, figsize=(8,12))
        m_on = axes[0].pcolormesh(X, Y, cog_on.T, norm=mpl.colors.LogNorm())
        plt.colorbar(m_on, ax=axes[0])
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].set_title('CoG of ON events on' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        m_off = axes[1].pcolormesh(X, Y, cog_off.T, norm=mpl.colors.LogNorm())
        plt.colorbar(m_off, ax=axes[1])
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Y [m]')
        axes[1].set_title('CoG of OFF events on' + args.date[:4] + '/' + args.date[4:6] + '/' + args.date[6:8])
        plt.savefig(path_dl2_data + '/cog-' + args.date + '.png')
        

