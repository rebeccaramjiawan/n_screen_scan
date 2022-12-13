#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class BTV to get deta from any BTV device

@author: fvelotti
"""
import numpy as np
import n_screen_scan.utils.frame_analysis as fa
from n_screen_scan.utils import Device
import time
import copy


def getParam_retry(func_alternative=time.sleep, **func_kwargs):
    def inner(func):
        def wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print("Error in getting parameter")
                    print(e)
                    func_alternative(**func_kwargs)

        return wrapper

    return inner


class BTV(Device):
    def __init__(
        self, japc, btv_name, plane="x", timingSelectorOverride=None
    ):
        super().__init__(japc, btv_name, timingSelectorOverride)

        self.btv_name = self.name

        if "BTV54" == self.btv_name:
            self.image_source = "BOVWA.11TCC4.AWAKECAM11/CameraImage"
            self.acquisition = "#image"

            self.height = "#height"
            self.width = "#width"
            self.p = "#pixelSize"

        elif "BTV42" == self.btv_name:
            self.image_source = "BOVWA.02TCV4.CAM9/CameraImage"
            self.acquisition = "#image"

            self.height = "#height"
            self.width = "#width"
            self.p = "#pixelSize"

        else:
            self.acquisition = "/Image#imageSet"
            self.axis1 = "/Image#imagePositionSet1"
            self.axis2 = "/Image#imagePositionSet2"
            self.height = "/Image#nbPtsInSet1"
            self.width = "/Image#nbPtsInSet2"

            self.image_source = self.name

        self.selector = (
            timingSelectorOverride
            if timingSelectorOverride is not None
            else self.japc.getSelector()
        )
        self.info = {}
        self.plane = plane

        self.image_all = None
        self._get_image_axis()
        self._ini_image = self._get_image()

        self.roi = self._calc_roi()

        self.frame_ana = fa.FrameAna(
            self._ini_image, self.im_ax1, self.im_ax2, self.roi
        )
        self.frame_ana.fit_gauss = True
        self.frame_ana.analyze_frame()

    def _calc_roi(self):

        pro_x = self._ini_image.sum(axis=0)
        pro_y = np.flip(self._ini_image.sum(axis=1))

        print(pro_x)
        print(pro_y)

        x = self.im_ax1
        y = self.im_ax2

        if "BTV54" in self.btv_name:
            mask_x = (x > -5) & (x < 5)
            mask_y = (y > -5) & (y < 0)
            x_min = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] - 4
            x_max = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] + 4
            y_min = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] - 4
            y_max = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] + 4
        elif "BTV42" in self.btv_name:
            mask_x = (x > -3) & (x < 4)
            mask_y = (y > -7) & (y < -1)
            x_min = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] - 2
            x_max = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] + 2
            y_min = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] - 2
            y_max = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] + 2
            # x_min = -10
            # x_max = 15
            # y_min = -20
            # y_max = 5
        elif "106" in self.btv_name:
            mask_x = (x > -5) & (x < 5)
            mask_y = (y > -5) & (y < 5)
            x_min = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] - 4
            x_max = self.im_ax1[mask_x][np.argmax(pro_x[mask_x])] + 4
            y_min = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] - 4
            y_max = self.im_ax2[mask_y][np.argmax(pro_y[mask_y])] + 4
        else:

            x_min = self.im_ax1[np.argmax(pro_x)] - 8
            x_max = self.im_ax1[np.argmax(pro_x)] + 8
            y_min = self.im_ax2[np.argmax(pro_y)] - 8
            y_max = self.im_ax2[np.argmax(pro_y)] + 8

        return np.array([x_min, x_max, y_min, y_max])

    @getParam_retry(secs=1.2)
    def _get_image(self, **kwds):
        self.valid_image = False
        for i in range(2):
            image, self.temp_info = self.japc.getParam(
                self.image_source + self.acquisition,
                timingSelectorOverride=self.selector,
                getHeader=True,
                **kwds
            )
            if "BOVWA" not in self.image_source:
                image = np.reshape(image, (self.w, self.h))

            self.valid_image = self._is_good_image(image)
            if self.valid_image:
                return image
            else:
                time.sleep(0.2)
        return image

    def _get_image_axis(self):
        if "BOVWA" in self.image_source:
            self.w = self.japc.getParam(
                self.image_source + self.height,
                timingSelectorOverride=self.selector,
            )
            self.h = self.japc.getParam(
                self.image_source + self.width,
                timingSelectorOverride=self.selector,
            )
            if self.btv_name == "BTV54":
                if self.acquisition == "#imageRawData":
                    self.pix_size = 0.134 / 5.0
                else:
                    self.pix_size = 0.134
            else:
                self.pix_size = (
                    self.japc.getParam(
                        self.image_source + self.p,
                        timingSelectorOverride=self.selector,
                    )
                    * 5.0
                )

        else:
            self.im_ax1 = self.japc.getParam(
                self.image_source + self.axis1,
                timingSelectorOverride=self.selector,
            )
            self.im_ax2 = self.japc.getParam(
                self.image_source + self.axis2,
                timingSelectorOverride=self.selector,
            )
            self.h = self.japc.getParam(
                self.image_source + self.height,
                timingSelectorOverride=self.selector,
            )
            self.w = self.japc.getParam(
                self.image_source + self.width,
                timingSelectorOverride=self.selector,
            )

        if "BOVWA" in self.image_source:
            self.im_ax1 = (
                np.linspace(-self.h / 2, self.h / 2, self.h)
                * self.pix_size
            )
            self.im_ax2 = (
                np.linspace(-self.w / 2, self.w / 2, self.w)
                * self.pix_size
            )

    def _get_multiple_images(
        self, nShot, delay_between_images=0.2, **kwargs
    ):
        self.image_all = None

        for count in range(nShot):
            self.raw_image = self._get_image(**kwargs)
            if "BOVWA" not in self.image_source:
                self.raw_image = np.reshape(
                    self.raw_image, (self.w, self.h)
                )

            self.frame_ana = fa.FrameAna()
            self.frame_ana.frame = self.raw_image
            self.frame_ana.x_ax = self.im_ax1
            self.frame_ana.y_ax = self.im_ax2

            self.frame_ana.roi = self.roi

            self.frame_ana.fit_gauss = True
            self.frame_ana.analyze_frame()

            if self.image_all is None:
                self.image_all = self.frame_ana.frame
            else:
                self.image_all = self.frame_ana.frame + self.image_all

            self.x_prof_array[count, :] = self.frame_ana.proj_x
            self.y_prof_array[count, :] = self.frame_ana.proj_y
            self.x_fit_array[count, :] = self.frame_ana.fit_x
            self.y_fit_array[count, :] = self.frame_ana.fit_y
            self.x_rms[count] = self.frame_ana.xRMS
            self.y_rms[count] = self.frame_ana.yRMS

            self.x_sigma[count] = self.frame_ana.sig_x
            self.y_sigma[count] = self.frame_ana.sig_y
            self.x_bar[count] = self.frame_ana.xBar
            self.y_bar[count] = self.frame_ana.yBar
            self.x_mu[count] = self.frame_ana.mean_x
            self.y_mu[count] = self.frame_ana.mean_y
            time.sleep(delay_between_images)

        self.image_all /= nShot
        return (
            np.mean(self.x_rms),
            np.mean(self.y_rms),
            np.mean(self.x_sigma),
            np.mean(self.y_sigma),
        )

    def _ini_data_array(self, nShot):
        self.x_prof_array = np.zeros(
            (nShot, self.frame_ana.frame.shape[1])
        )
        self.y_prof_array = np.zeros(
            (nShot, self.frame_ana.frame.shape[0])
        )

        self.x_fit_array = np.zeros(
            (nShot, self.frame_ana.frame.shape[1])
        )
        self.y_fit_array = np.zeros(
            (nShot, self.frame_ana.frame.shape[0])
        )

        self.x_rms = np.zeros(nShot)
        self.y_rms = np.zeros(nShot)

        self.x_bar = np.zeros(nShot)
        self.y_bar = np.zeros(nShot)

        self.x_sigma = np.zeros(nShot)
        self.y_sigma = np.zeros(nShot)

        self.x_mu = np.zeros(nShot)
        self.y_mu = np.zeros(nShot)

    def _is_good_image(self, image):
        p_x = image.sum(axis=0)
        p_y = image.sum(axis=1)

        mean_x = np.median(p_x)
        mean_y = np.median(p_y)

        if "BTV54" in self.btv_name:
            try:
                result = (
                    np.max(p_x) > 1.8 * mean_x
                    and np.max(p_y) > 1.8 * mean_y
                )
            except:
                result = False
        else:
            try:
                result = (
                    np.max(p_x) > 1.5 * mean_x
                    or np.max(p_y) > 1.5 * mean_y
                )
            except:
                result = False
        return result

    def _make_bi_gauss_fit(self):
        # TODO: implement double Gaussian fit
        pass

    def _gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height * np.exp(
            -(
                ((center_x - x) / width_x) ** 2
                + ((center_y - y) / width_y) ** 2
            )
            / 2
        )

    def _moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments"""
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        width_x = np.sqrt(
            np.abs((np.arange(col.size) - y) ** 2 * col).sum()
            / col.sum()
        )
        row = data[int(x), :]
        width_y = np.sqrt(
            np.abs((np.arange(row.size) - x) ** 2 * row).sum()
            / row.sum()
        )
        height = data.max()
        return height, x, y, width_x, width_y

    def get_image_data(self, rms=False, nShot=1, **kwargs):
        self._ini_image = self._get_image(**kwargs)
        self.info = copy.deepcopy(self.temp_info)

        # self._get_image_axis()
        self.roi = self._calc_roi()

        self.frame_ana = fa.FrameAna(
            self._ini_image, self.im_ax1, self.im_ax2, self.roi
        )
        self.frame_ana.fit_gauss = True
        self.frame_ana.analyze_frame()

        self._ini_data_array(nShot)

        xr, yr, xs, ys = self._get_multiple_images(nShot, **kwargs)

        data_btv_plotting = {
            "x": self.frame_ana.x_ax,
            "pro_x": np.mean(self.x_prof_array, axis=0),
            "y": self.frame_ana.y_ax,
            "pro_y": np.mean(self.y_prof_array, axis=0),
            "fit_x": np.mean(self.x_fit_array, axis=0),
            "fit_y": np.mean(self.y_fit_array, axis=0),
            "rms_x": self.x_rms,
            "rms_y": self.y_rms,
            "sigma_x": self.x_sigma,
            "sigma_y": self.y_sigma,
            "mu_x": np.mean(self.x_mu),
            "mu_y": np.mean(self.y_mu),
            "image": self.image_all,
            "raw_image": self.raw_image,
        }

        if rms:
            return xr, yr, data_btv_plotting
        else:
            return xs, ys, data_btv_plotting

    def getParameter(self, getHeader=True, **kwrds):
        # TODO: Check the unitis returned => it should be mm
        _, _, data_all = self.get_image_data(**kwrds)
        info = self.info if getHeader else {}
        return data_all, info

    def getPositon(self, getHeader=True, **kwargs):

        data_all, info = self.getParameter(
            getHeader=getHeader, **kwargs
        )
        return data_all[self.plane], info
