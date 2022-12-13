#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:30:02 2017

@author: sgess
"""

import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from numpy.fft import fft

''' Class for analyzing images '''


class FrameAna(object):
    def __init__(self, frame=[], x_ax=[], y_ax=[], roi=[]):
        self.init_frame = frame
        self.init_xax = x_ax
        self.init_yax = y_ax
        self.frame = frame
        self.x_ax = x_ax
        self.y_ax = y_ax
        self.roi = roi
        self.bg_frame = np.array([])
        self.bg_sub = False
        self.median_filter = False
        self.fit_gauss = False
        self.streak_width = 2.5
        self.marker_lim = -3.5
        self.nEmbed = 4096
        self.frame_analyzed = False

    ''' Main image analysis method '''

    def analyze_frame(self):

        # Find pixel sum of image before applying manipulations
        self.sum_all = self.frame.sum()

        # Cast image as float and apply median filter
        self.frame = self.frame.astype(float)

        # Subtract background image
        if self.bg_sub:
            if self.bg_frame.any():
                if np.size(self.bg_frame) != np.size(self.frame):
                    print('Warning: Background frame size does not match image frame size. Background not subtracted')
                else:
                    self.frame = self.frame - self.bg_frame

        # Apply median filter
        if self.median_filter:
            self.frame = sig.medfilt2d(self.frame)

        # Extract ROI and relevant portions of axes
        self.x_ind = (self.x_ax >= self.roi[0]) & (self.x_ax <= self.roi[1])
        self.y_ind = (self.y_ax >= self.roi[2]) & (self.y_ax <= self.roi[3])
        # self.y_ind = np.flipud(self.y_ind)
        # print(self.y_ind.shape)
        self.frame = np.flipud(self.frame)
        self.frame = self.frame[np.ix_(self.y_ind, self.x_ind)]

        # print
        self.x_ax = self.x_ax[self.x_ind]
        # self.y_ind = np.flipud(self.y_ind)
        self.y_ax = self.y_ax[self.y_ind]
        # print(self.y_ax[0])
        # print(self.y_ax[-1])

        # Get frame min and max
        self.min = np.amin(self.frame)
        self.max = np.amax(self.frame)

        # Find pixel sum and projections of ROI'd image
        self.sum_proc = self.frame.sum()
        self.proj_x = self.frame.mean(0)
        self.proj_y = self.frame.mean(1)

        # Find centroid and RMS
        self.xBar, self.xRMS = self.profile_moments('x')
        self.yBar, self.yRMS = self.profile_moments('y')
        self.xMax = np.max(self.proj_x)
        self.xMin = np.min(self.proj_x)
        self.yMax = np.max(self.proj_y)
        self.yMin = np.min(self.proj_y)

        # Perform Gaussian fits
        if self.fit_gauss:
            self.gaussFit('x')
            self.gaussFit('y')

        self.frame_analyzed = True
        self.frame = np.flipud(self.frame)

    ''' Streak Analysis Method '''

    def streak_ana(self):

        if not self.frame_analyzed:
            print('Must run analyze_frame first.')
            return

        # Extract axes centroid and image
        self.t_ax = np.linspace(self.y_ax[0], self.y_ax[-1], len(self.y_ax))
        # self.t_ax = self.y_ax

        # Create lineout and bands for FFT
        if self.fit_gauss:
            xc = self.mean_x
            xs = self.sig_x
        else:
            xc = self.xBar
            xs = self.xRMS

        # xl is lower band, xh, is the upper band, and xc is the center line
        xl = xc - xs * self.streak_width
        xh = xc + xs * self.streak_width
        x_ind = (self.x_ax > xl) & (self.x_ax < xh)
        c_ind = np.argmin(min(abs(self.x_ax - xc)))

        # Orig profs
        oBand = self.frame[:, x_ind].mean(1)
        oLine = self.frame[:, c_ind]
        oOutb = self.frame[:, ~x_ind].mean(1)

        # New interpolation
        band = np.interp(self.t_ax, self.y_ax, oBand)
        line = np.interp(self.t_ax, self.y_ax, oLine)
        outb = np.interp(self.t_ax, self.y_ax, oOutb)

        hann_band = np.hanning(len(band)) * band
        hann_line = np.hanning(len(line)) * line
        hann_outb = np.hanning(len(outb)) * outb

        # Create FFT axis
        nsamp = len(self.t_ax)
        s_max = round(nsamp / 2)
        dt = self.t_ax[1] - self.t_ax[0]
        # dt = np.mean(np.diff(self.t_ax))
        f_max = 1 / dt
        full_ax = np.linspace(0, f_max, nsamp)
        f_ax = 1000 * full_ax[0:s_max]

        # Create FFT embed
        arb = self.nEmbed
        embed = np.zeros(arb)
        embed[(round(arb / 2) - round(nsamp / 2)):(round(arb / 2) + round(nsamp / 2))] = hann_band
        pad_ax = np.linspace(0, f_max, arb)
        f_pad = 1000 * pad_ax[0:round(arb / 2)]

        # FFT the data
        fb = fft(band)
        fl = fft(line)
        fo = fft(outb)

        hb = fft(hann_band)
        hl = fft(hann_line)
        ho = fft(hann_outb)
        he = fft(embed)

        # Get the absolute value of the FFT
        fftb = abs(fb[0:s_max])
        fftl = abs(fl[0:s_max])
        ffto = abs(fo[0:s_max])

        hftb = abs(hb[0:s_max])
        hftl = abs(hl[0:s_max])
        hfto = abs(ho[0:s_max])
        hfte = abs(he[0:round(arb / 2)])

        # Store the results
        self.inner_band = band
        self.center_line = line
        self.outer_band = outb
        self.hann_band = hann_band
        self.hann_line = hann_line
        self.hann_outb = hann_outb
        self.f_ax = f_ax
        self.band_fft = fftb
        self.line_fft = fftl
        self.outer_fft = ffto
        self.band_hft = hftb
        self.line_hft = hftl
        self.outer_hft = hfto
        self.embed_hft = hfte
        self.f_pad = f_pad
        self.band_ref = [xl, xc, xh]

    ''' Streak Analysis Method '''

    def marker_ana(self):

        if not self.frame_analyzed:
            print('Must run analyze_frame first.')
            return
        new_frame = np.array(self.init_frame.astype(float))
        self.m_ind = (self.init_xax < self.marker_lim)
        marker_area = new_frame[np.ix_(self.y_ind, self.m_ind)]
        mark_area = sig.medfilt2d(marker_area)
        self.m_ax = np.linspace(self.y_ax[0], self.y_ax[-1], len(self.y_ax))
        self.proj_m = np.interp(self.m_ax, self.y_ax, mark_area.mean(1))
        self.mark_ind = np.argmax(self.proj_m)
        self.mark_val = self.y_ax[self.mark_ind]
        self.gaussFit('m')

    ''' Generate COGs. Default Function '''

    def profile_moments(self, axis):

        if axis == 'x':
            no_zero = self.proj_x - min(self.proj_x)
            cent = no_zero.dot(self.x_ax) / no_zero.sum()
            rms = np.sqrt(no_zero.dot((self.x_ax - cent) ** 2) / no_zero.sum())
        elif axis == 'y':
            no_zero = self.proj_y - min(self.proj_y)
            cent = no_zero.dot(self.y_ax) / no_zero.sum()
            rms = np.sqrt(no_zero.dot((self.y_ax - cent) ** 2) / no_zero.sum())

        return cent, rms

    ''' Define Gaussian Shape '''

    def gaussian(self, x, amp, cen, wid, off):

        return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + off

    ''' Fit Gaussian. Not called by default '''

    def gaussFit(self, axis):

        if axis == 'x':
            guess = [self.xMax - self.xMin, self.xBar, self.xRMS, self.xMin]
            # print(guess)
            try:
                result, pcov = curve_fit(self.gaussian, self.x_ax, self.proj_x, guess)
                result[2] = abs(result[2])
                fit = self.gaussian(self.x_ax, *result)
            except:
                print('Failed to fit in ' + axis + '-direction')
                result = [0, 0, 0, 0]
                fit = np.zeros(np.shape(self.x_ax))

            self.amp_x = result[0]
            self.mean_x = result[1]
            self.sig_x = result[2]
            self.off_x = result[3]
            self.fit_x = fit

        elif axis == 'y':
            guess = [self.yMax - self.yMin, self.yBar, self.yRMS, self.yMin]
            # print(guess)
            try:
                result, pcov = curve_fit(self.gaussian, self.y_ax, self.proj_y, guess)
                result[2] = abs(result[2])
                fit = self.gaussian(self.y_ax, *result)
            except:
                print('Failed to fit in ' + axis + '-direction')
                result = [0, 0, 0, 0]
                fit = np.zeros(np.shape(self.y_ax))

            self.amp_y = result[0]
            self.mean_y = result[1]
            self.sig_y = result[2]
            self.off_y = result[3]
            self.fit_y = fit

        elif axis == 'm':
            guess = [np.max(self.proj_m) - np.min(self.proj_m), self.mark_val, 10, np.min(self.proj_m)]
            # print(guess)
            try:
                result, pcov = curve_fit(self.gaussian, self.y_ax, self.proj_m, guess)
                result[2] = abs(result[2])
                fit = self.gaussian(self.y_ax, *result)
            except:
                print('Failed to fit in ' + axis + '-direction')
                result = [0, 0, 0, 0]
                fit = np.zeros(np.shape(self.y_ax))

            self.amp_m = result[0]
            self.mean_m = result[1]
            self.sig_m = result[2]
            self.off_m = result[3]
            self.fit_m = fit
