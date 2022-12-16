import numpy as np
import os
from cpymad.madx import Madx
from scipy.optimize import curve_fit
import pickle
import git
import datetime
import json
from n_screen_scan.utils.myjson import myJSONEncoder

class BTV_calc:
    def __init__(self, energy, mass, emitta_x, emitta_y, mom_spr, btv_madx, btvs, btv_screens, simulate, particle, n_btv_x,
                 n_btv_y):
        self.madx = Madx(stdout=False)
        self.energy = energy
        self.mass = mass
        self.gamma = self.energy/self.mass # approximation to the lorentz gamma
        self.beta = np.sqrt(1-np.divide(1, np.square(self.gamma))) # lorentz beta
        self.emitta_x = emitta_x # horizontal emittance in um
        self.emitta_y = emitta_y # vertical emittance in um
        self.mom_spr = mom_spr # momentum spread
        self.btv = btv_madx
        self.btvs = btvs
        self.btv_screens = btv_screens
        self.simulate = simulate # to use simulated or real data
        self.particle = particle
        self.n_btv_x = n_btv_x # number of horizontal btvs
        self.n_btv_y = n_btv_y # number of vertical btvs
        self.bad_data = False
        # clone_dir = 'n_screen_scan\madx_cloned'

        ## Get up-to-date beamline sequence from Git (can only use from CERN)
        # if os.path.exists(clone_dir):
        #     pass
        # else:
        #     print("making directory... " + clone_dir)
        #     os.mkdir(clone_dir)
        #     git.Git(clone_dir).clone("https://gitlab.cern.ch/acc-models/acc-models-tls.git", depth=1)

        ## Call madx files (if not on-site at CERN)
        current_dir = os.getcwd()
        # os.chdir(clone_dir + '/acc-models-tls/awake_injection/tt43/line')
        os.chdir(current_dir + '\\n_screen_scan\madx\\')
        self.madx.call(file=os.getcwd() + '\\general_tt43.madx')
        self.madx.call(file=os.getcwd() + '\\tt43.seq')
        self.madx.call(file=os.getcwd() + '\\str/focus_btv54.str')
        os.chdir(current_dir)
        sequence_name = self.madx.sequence()

        # Define guess of initial beam parameters from model params
        variables = self.madx.globals
        self.betx0 = variables['betx0']
        self.bety0 = variables['bety0']
        self.alfx0 = variables['alfx0']
        self.alfy0 = variables['alfy0']
        self.dx0 = variables["dx0"]
        self.dy0 = variables["dy0"]
        self.dpx0 = variables["dpx0"]
        self.dpy0 = variables["dpy0"]

        # Set up sequence and strengths of magnets
        self.sequence_name = str(sequence_name)[11:-1]
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        self.madx.use(sequence=self.sequence_name)
        self.madx.select(flag='RMATRIX')
        self.madx.input("DeltaP = " + str(self.mom_spr) + ";")
        self.madx.input("emitta_x = " + str(self.emitta_x) + ";")
        self.madx.input("emitta_y = " + str(self.emitta_y) + ";")
        self.madx.input(
            "sigma_x := (sqrt((table(twiss, betx)*emitta_x) + ((table(twiss,dx))*DeltaP)*((table(twiss,dx))*DeltaP)));")
        self.madx.input(
            "sigma_y :=  (sqrt((table(twiss, bety)*emitta_y) + ((table(twiss,dy))*DeltaP)*((table(twiss,dy))*DeltaP)));")

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0,
                                RMATRIX=True, file='data/twiss_test.out')
        btv_ind = [any(b in s.lower() for b in self.btv) for s in twiss["name"]]
        self.disp_x = (twiss['dx'][btv_ind])
        self.disp_y = (twiss['dy'][btv_ind])

    def r_val_calc(self, btv1, *args):
        # Calculate the R-matrix parameters which quantify the beam propagation in 6D
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'alfx', 'alfy', 'dx',
                                 'dy', 'mux', 'muy', 'RE11', 'RE12', 'RE33', 'RE34'])
        if len(args) == 0:
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                    ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0,
                                    RMATRIX=True, range=btv1)
            r11 = twiss["RE11"][twiss["name"] == btv1 + ":1"][0]
            r12 = 0 * twiss["RE12"][twiss["name"] == btv1 + ":1"][0]
            r21 = 0 * twiss["RE21"][twiss["name"] == btv1 + ":1"][0]
            r22 = twiss["RE22"][twiss["name"] == btv1 + ":1"][0]
            r33 = twiss["RE33"][twiss["name"] == btv1 + ":1"][0]
            r34 = 0 * twiss["RE34"][twiss["name"] == btv1 + ":1"][0]
            r43 = 0 * twiss["RE43"][twiss["name"] == btv1 + ":1"][0]
            r44 = twiss["RE44"][twiss["name"] == btv1 + ":1"][0]
        elif len(args) == 1:
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                    ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0,
                                    RMATRIX=True, range=btv1 + "/" + args[0])
            r11 = twiss["RE11"][twiss["name"] == args[0] + ":1"][0]
            r12 = twiss["RE12"][twiss["name"] == args[0] + ":1"][0]
            r33 = twiss["RE33"][twiss["name"] == args[0] + ":1"][0]
            r34 = twiss["RE34"][twiss["name"] == args[0] + ":1"][0]
            r22 = twiss["RE22"][twiss["name"] == args[0] + ":1"][0]
            r21 = twiss["RE21"][twiss["name"] == args[0] + ":1"][0]
            r44 = twiss["RE44"][twiss["name"] == args[0] + ":1"][0]
            r43 = twiss["RE43"][twiss["name"] == args[0] + ":1"][0]
        else:
            print("Wrong number of input arguments to r_val_calc")
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                    ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0,
                                    RMATRIX=True, range=btv1 + "/" + args[0])
            r11 = twiss["RE11"][twiss["name"] == args[0] + ":1"][0]
            r12 = twiss["RE12"][twiss["name"] == args[0] + ":1"][0]
            r33 = twiss["RE33"][twiss["name"] == args[0] + ":1"][0]
            r34 = twiss["RE34"][twiss["name"] == args[0] + ":1"][0]
            r22 = twiss["RE22"][twiss["name"] == args[0] + ":1"][0]
            r21 = twiss["RE21"][twiss["name"] == args[0] + ":1"][0]
            r44 = twiss["RE44"][twiss["name"] == args[0] + ":1"][0]
            r43 = twiss["RE43"][twiss["name"] == args[0] + ":1"][0]
        return r11, r12, r33, r34, r21, r22, r43, r44

    def _rmat_calc(self, dim):
        # Collate the R-matrix parameters between different BTVs
        btv = self.btv
        rmat = []
        if dim == "x":
            diff_disp = (self.disp_x - self.disp_x[0])
            for counter, i in enumerate(btv[0:self.n_btv_x]):
                if counter == 0:
                    r11, r12, r33, r34, _, _, _, _ = self.r_val_calc(btv[0])
                    rmat = [r11 * r11, -2 * r11 * r12, r12 * r12, 2 * r11 * diff_disp[counter],
                            2 * r12 * diff_disp[counter], diff_disp[counter] ** 2]
                else:
                    r11, r12, r33, r34, _, _, _, _ = self.r_val_calc(btv[0], i)
                    rmat = np.vstack((rmat, [r11 * r11, -2 * r11 * r12, r12 * r12, 2 * r11 * diff_disp[counter],
                                             2 * r12 * diff_disp[counter], diff_disp[counter] ** 2]))
        elif dim == "y":
            diff_disp = (self.disp_y - self.disp_y[0])
            for counter, i in enumerate(btv[0:self.n_btv_y]):
                if counter == 0:
                    r11, r12, r33, r34, _, _, _, _ = self.r_val_calc(btv[0])
                    rmat = [r33 * r33, -2 * r33 * r34, r34 * r34, 2 * r33 * diff_disp[counter],
                            2 * r34 * diff_disp[counter], diff_disp[counter] ** 2]
                else:
                    r11, r12, r33, r34, _, _, _, _ = self.r_val_calc(btv[0], i)
                    rmat = np.vstack((rmat, [r33 * r33, -2 * r33 * r34, r34 * r34, 2 * r33 * diff_disp[counter],
                                             2 * r34 * diff_disp[counter], diff_disp[counter] ** 2]))
        if dim == "x":
            if self.n_btv_x == 4:
                eta = np.reshape(self.disp_x, (len(self.disp_x), 1))
                rmat = np.append(rmat[:4, :3], eta[:4, :], axis=1)
            # elif self.n_btv_x == 6:
        if dim == "y":
            if self.n_btv_y == 4:
                eta = np.reshape(self.disp_y, (len(self.disp_y), 1))
                rmat = np.append(rmat[:4, :3], eta[:4, :], axis=1)
        return rmat

    def _btv_calc(self, rmat, sigma, disp):
        # Fit to the data to estimate beam parameters, more BTVs mean more
        # params you can estimate because dimensionality must match or underconstrained

        epsilon = beta = gamma = alpha = dispersion = deriv_dispersion = delta = 0
        if len(sigma) == 3:
            b = sigma
            q = np.dot(np.linalg.inv(rmat[:3, :3]), b)
            # convert output of matrix multiplication (q) to Twiss parameters and momentum spread
            gamma_r = np.sqrt(self.energy ** 2 + self.mass ** 2) / self.mass
            epsilon = np.sqrt(np.abs((q[0] * q[2]) - (q[1] * q[1])))
            beta = q[0] / epsilon
            alpha = q[1] / epsilon
            gamma = q[2] / epsilon
            print('epsilon, beta, alpha')
            print(epsilon * gamma_r, beta, alpha)
        elif len(sigma) == 4:
            b = sigma
            q = np.dot(np.linalg.inv(rmat[:4, :4]), b)
            # convert output of (q) to Twiss parameters and momentum spread
            gamma_r = np.sqrt(self.energy ** 2 + self.mass ** 2) / self.mass
            epsilon = np.sqrt(np.abs((q[0] * q[2]) - (q[1] * q[1])))
            beta = q[0] / epsilon
            alpha = q[1] / epsilon
            gamma = q[2] / epsilon
            momspr = np.sqrt(np.abs(q[3])) * 10 ** -3
            print('epsilon, beta, alpha, momentum spread')
            print(epsilon * gamma_r, beta, alpha, momspr)
        elif len(sigma) == 5:
            diff_disp = (disp - disp[0])
            b = sigma * 10 ** -6 - np.square(np.multiply(diff_disp, self.mom_spr))
            # rmat = np.append(rmat, eta, axis=1)
            q = np.dot(np.linalg.inv(rmat[:5, :5]), b)
            # convert output of (q) to Twiss parameters and momentum spread
            gamma_r = np.sqrt(self.energy ** 2 + self.mass ** 2) / self.mass
            a = q[0] - (q[3] ** 2) / (self.mom_spr ** 2)
            b = q[1] + (q[3] * q[4]) / (self.mom_spr ** 2)
            c = q[2] - (q[4] ** 2) / (self.mom_spr ** 2)
            epsilon = np.sqrt(np.abs((a * c) - (b ** 2)))
            beta = a / epsilon
            alpha = b / epsilon
            gamma = q[2] / epsilon
            dispersion = q[3] / self.mom_spr ** 2
            deriv_dispersion = q[4] / self.mom_spr ** 2
            epsilon *= 10 ** 6
            print('epsilon, beta, alpha, momentum spread, dispersion, derivative of dispersion')
            print(epsilon * gamma_r, beta, alpha, dispersion, deriv_dispersion)
        elif len(sigma) == 6:
            b = sigma * 10 ** -6
            q = np.dot(np.linalg.inv(rmat), b)
            # convert output of fit (q) to Twiss parameters and momentum spread
            gamma_r = np.sqrt(self.energy ** 2 + self.mass ** 2) / self.mass
            delta = np.sqrt(abs(q[5])) * 0.1
            a = q[0] - (q[3] ** 2) / (delta ** 2)
            b = q[1] + (q[3] * q[4]) / (delta ** 2)
            c = q[2] - (q[4] ** 2) / (delta ** 2)
            epsilon = np.sqrt(np.abs((a * c) - (b ** 2)))
            beta = a / epsilon
            alpha = b / epsilon
            gamma = q[2] / epsilon
            dispersion = q[3] / delta ** 2
            deriv_dispersion = q[4] / delta ** 2
            epsilon *= 10 ** 6
            print('epsilon, beta, alpha, dispersion, derivative of dispersion, mom spr')
            print(epsilon * gamma_r, beta, alpha, dispersion, deriv_dispersion, delta)
        elif len(sigma) > 6:
            b = np.squeeze(sigma) * 10 ** -6
            q = np.linalg.lstsq(rmat, b, rcond=None)[0]
            # convert output of fit (q) to Twiss parameters and momentum spread
            gamma_r = np.sqrt(self.energy ** 2 + self.mass ** 2) / self.mass
            delta = np.sqrt(abs(q[5])) * 10 ** 3
            a = q[0] - (q[3] ** 2) / (delta ** 2)
            b = q[1] + (q[3] * q[4]) / (delta ** 2)
            c = q[2] - (q[4] ** 2) / (delta ** 2)
            epsilon = np.sqrt(np.abs((a * c) - (b ** 2)))
            beta = a / epsilon
            alpha = b / epsilon
            gamma = q[2] / epsilon
            dispersion = q[3] / delta ** 2
            deriv_dispersion = q[4] / delta ** 2
            epsilon *= 10 ** 6
            print('epsilon, beta, alpha, momentum spread, dispersion, derivative of dispersion, mom spr')
            print(epsilon * gamma_r, beta, alpha, dispersion, deriv_dispersion, delta * 10 ** -3)
        return epsilon, beta, alpha, gamma, dispersion, deriv_dispersion, delta * 10 ** -3

    def emit_calc(self, n_shot, is_rms):
        # Calculate and compare expected emittance with calculated emittance

        if self.simulate is True:
            sigma_x, sigma_y, data_all = self._fake_get_data(n_shot)
        else:
            sigma_x, sigma_y, data_all = self._get_data(n_shot, is_rms)
        rmat_x = self._rmat_calc("x") # Beam propagation matrix - horizontal
        rmat_y = self._rmat_calc("y") # Beam propagation matrix - vertical
        print("btv calc x")
        epsilon_x, beta_x, alpha_x, gamma_x, dispersion_x, deriv_dispersion_x, delta_x = self._btv_calc(
            rmat_x[0:self.n_btv_x, 0:self.n_btv_x],
            sigma_x[0:self.n_btv_x],
            self.disp_x[0:self.n_btv_x])
        print("btv calc y")
        epsilon_y, beta_y, alpha_y, gamma_y, dispersion_y, deriv_dispersion_y, delta_y = self._btv_calc(
            rmat_y[0:self.n_btv_y, 0:self.n_btv_y],
            sigma_y[0:self.n_btv_y],
            self.disp_y[0:self.n_btv_y])
        print("------------------------------------")
        if self.simulate is True:
            print("Input emittance [x] = " + str(np.round(self.emitta_x * self.energy * 10 ** 6 / self.mass, 2)) + " um")
            print("Input emittance [y] = " + str(np.round(self.emitta_y * self.energy * 10 ** 6 / self.mass, 2)) + " um")
        print("Estimated emittance [x] = " + str(np.round(epsilon_x * self.energy / self.mass, 2)) + " um")
        print("Estimated emittance [y] = " + str(np.round(epsilon_y * self.energy / self.mass, 2)) + " um")
        estimates = {"epsilon_x": epsilon_x, "epsilon_y": epsilon_y, "beta_x": beta_x, "beta_y": beta_y,
                     "alpha_x": alpha_x, "alpha_y": alpha_y, "gamma_x": gamma_x, "gamma_y": gamma_y,
                     "dispersion_x": dispersion_x*self.beta, "dispersion_y": dispersion_y*self.beta,
                     "deriv_dispersion_x": deriv_dispersion_x*self.beta, "deriv_dispersion_y": deriv_dispersion_y*self.beta,
                     "delta_x": delta_x, "delta_y": delta_y}
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0,
                                RMATRIX=True)
        self._json_data(estimates, data_all)
        return estimates, data_all, twiss

    def _get_data(self, n_shot, is_rms):
        # import pyjapc
        # japc = pyjapc.PyJapc(noSet=True)
        # japc.rbacLogin()
        # btv_screens = self.btv_screens
        # t0 = time.time()

        # data_all = {}
        #
        # rms_x = np.zeros(len(btvs))
        # rms_y = np.zeros(len(btvs))
        #
        # # Insert first three BTVs
        # # To find settings under FESA, tick box and click i for information
        # for btv in btvs_screens:
        #     #	japc.setParam(btv +'/MotorEnable#enable', 1, timingSelectorOverride='')
        #     # Do these with paramaters_utils?
        #     japc.setParam(btv + '/Setting#cameraSwitch', 3, timingSelectorOverride='')
        #     japc.setParam(btv + '/Setting#screenSelect', 1, timingSelectorOverride='')
        #     time.sleep(4)
        #     japc.setParam(btv + '/Setting#filterSelect', 2, timingSelectorOverride='')
        #     time.sleep(4)
        #
        # # Check all required screens are inserted
        #
        # for i, btv in enumerate(btvs):
        #     print(btv)
        #     btv_obj = BTV(japc=japc, btv_name=btv)
        #     if is_rms is True:
        #         rms = "rms"
        #     else:
        #         rms = "std"
        #     rms_x[i], rms_y[i], data_btv = btv_obj.get_image_data(n_shot=n_shot, rms=rms)
        #     data_all[btv] = data_btv
        #     data_all[btv]['rms_x'] = rms_x[i]
        #     data_all[btv]['rms_y'] = rms_y[i]
        #     # if not last btv, remove btv after taking data
        #     if self.particle == "p":
        #             #  pass
        #             # Remove BTV
        #             # japc.setParam(btvs_screens[i] + '/Setting#screenSelect', 0, timingSelectorOverride='')
        #             # time.sleep(4)
        #     elif self.particle == "e":
        #         if i < (len(btvs) - 1):
        #             # Remove BTV
        #             japc.setParam(btvs_screens[i] + '/Setting#screenSelect', 0, timingSelectorOverride='')
        #             time.sleep(4)
        #     else:
        #         print("Particle type not set")
        #
        # t1 = time.time()
        #
        # print('time enlapsed = ', t1 - t0)

        #  If not at CERN i.e. can't collect data, can fake the data
        self.madx.use(SEQUENCE=self.sequence_name, range="#s/plasma_merge")
        rms_x = np.empty(shape=(len(self.btvs),))
        rms_y = np.empty(shape=(len(self.btvs),))

        file = open("data/data_4screen_2020-06-26_12_24_29.563667.p", "rb")
        object_file = pickle.load(file)
        file.close()

        for counter, screen in enumerate(self.btvs):
            rms_x[counter] = np.square(object_file[screen]['rms_x'])
            rms_y[counter] = np.square(object_file[screen]['rms_y'])
            data_all = object_file
        return rms_x, rms_y, data_all

    def _fake_get_data(self, n_shot):
        # Initialise
        data_all = dict()
        sigma_x = np.zeros((n_shot, len(self.btv)))
        sigma_y = np.zeros((n_shot, len(self.btv)))
        mu_x = np.zeros((n_shot, len(self.btv)))
        mu_y = np.zeros((n_shot, len(self.btv)))
        btvs = self.btvs

        # xmin=-0.003 * np.ones(shape=(len(self.btv), 1))
        # xmax = 0.003 * np.ones(shape=(len(self.btv), 1))
        # ymin = -0.003 * np.ones(shape=(len(self.btv), 1))
        # ymax = 0.003 * np.ones(shape=(len(self.btv), 1))

        # Dimensions of BTV for plotting - work on automating this
        xmin = [[-0.003], [-0.003], [-0.006], [-0.002]]
        xmax = [[0.003], [0.003], [0.006], [0.002]]
        ymin = [[-0.003], [-0.003], [-0.006], [-0.002]]
        ymax = [[0.003], [0.003], [0.006], [0.002]]

        # Initialise
        for i, btv in enumerate(self.btv):
            data_all[btvs[i]] = dict()
            data_all[btvs[i]] = {"image": None}

        # Extract beam parameters from 2D image data
        for j in range(n_shot):
            ptc_output = self.track()
            for i, btv in enumerate(self.btv):
                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                        ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0)
                s = twiss['s'][twiss["name"] == (btv + ":1")][0]
                tracking_data_x = ptc_output["x"][ptc_output["s"] == s]
                tracking_data_y = ptc_output["y"][ptc_output["s"] == s]
                if data_all[btvs[i]]["image"] is None:
                    data_all[btvs[i]]["image"] = \
                        np.histogram2d(tracking_data_x, tracking_data_y, bins=[79, 150],
                                       range=[[xmin[i][0], xmax[i][0]], [ymin[i][0], ymax[i][0]]])[0]
                    data_temp = np.histogram2d(tracking_data_x, tracking_data_y, bins=[79, 150],
                                               range=[[xmin[i][0], xmax[i][0]], [ymin[i][0], ymax[i][0]]])[0]
                else:
                    data_temp = np.histogram2d(tracking_data_x, tracking_data_y, bins=[79, 150],
                                               range=[[xmin[i][0], xmax[i][0]], [ymin[i][0], ymax[i][0]]])[0]
                    data_all[btvs[i]]["image"] = data_all[btvs[i]]["image"] + data_temp
                proj = np.sum(data_temp, axis=1)

                x_ax = np.linspace(xmin[i][0] * 10 ** 3, xmax[i][0] * 10 ** 3, len(proj))
                no_zero = proj - min(proj)
                cent = no_zero.dot(x_ax) / no_zero.sum()
                rms = np.sqrt(no_zero.dot((x_ax - cent) ** 2) / no_zero.sum())
                guess = [np.max(proj) - np.min(proj), cent, rms, np.min(proj)]
                try:
                    popt_x, popt_x_err = curve_fit(self.gaussian, x_ax, proj, guess)
                except:
                    print("x " + str(btv) + " failed to fit")
                    popt_x = guess

                proj = np.sum(data_temp, axis=0)

                y_ax = np.linspace(ymin[i][0] * 10 ** 3, ymax[i][0] * 10 ** 3, len(proj))
                no_zero = proj - min(proj)
                cent = no_zero.dot(y_ax) / no_zero.sum()
                rms = np.sqrt(no_zero.dot((y_ax - cent) ** 2) / no_zero.sum())
                guess = [np.max(proj) - np.min(proj), cent, rms, np.min(proj)]
                try:
                    popt_y, popt_y_err = curve_fit(self.gaussian, y_ax, proj, guess)
                except:
                    print("y " + str(btv) + " failed to fit")
                    popt_y = guess
                sigma_x[j, i] = popt_x[2]
                sigma_y[j, i] = popt_y[2]
                mu_x[j, i] = popt_x[1]
                mu_y[j, i] = popt_y[1]
                data_all[btvs[i]]["fit_x"] = self.gaussian(x_ax, *popt_x)
                data_all[btvs[i]]["fit_y"] = self.gaussian(y_ax, *popt_y)

        # Collate together for all BTVs
        for i, btv in enumerate(self.btvs):
            data_all[btv]["image"] /= n_shot
            data_all[btv]["pro_x"] = np.sum(data_all[btv]["image"], axis=1)
            data_all[btv]["pro_y"] = np.sum(data_all[btv]["image"], axis=0)
            data_all[btv]["x"] = np.linspace(xmin[i][0], xmax[i][0], len(data_all[btv]["pro_x"])) * 10 ** 3
            data_all[btv]["y"] = np.linspace(ymin[i][0], ymax[i][0], len(data_all[btv]["pro_y"])) * 10 ** 3
            data_all[btv]["sigma_x"] = np.mean(sigma_x[:, i], axis=0)
            data_all[btv]["sigma_y"] = np.mean(sigma_y[:, i], axis=0)
            data_all[btv]["mu_x"] = np.average(mu_x[:, i], axis=0)
            data_all[btv]["mu_y"] = np.average(mu_y[:, i], axis=0)
            data_all[btv]["amp"] = np.max(data_all[btv]["image"])
            data_all[btv]['err_x'] = np.average(sigma_x[:, i], axis=0)/np.sqrt(n_shot)
            data_all[btv]['err_y'] = np.average(sigma_y[:, i], axis=0)/np.sqrt(n_shot)
            data_all[btv]['mu_err_x'] = np.average(sigma_x[:, i], axis=0)/np.sqrt(2*n_shot)
            data_all[btv]['mu_err_y'] = np.average(sigma_y[:, i], axis=0)/np.sqrt(2*n_shot)
        return np.average(sigma_x, axis=0), np.average(sigma_y, axis=0), data_all

    def gaussian(self, x, amp, cen, wid, off):
        return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + off

    def track(self):
        # Track beam through simulation of beamline, set number of particles below
        seed = 42
        sigmas = 3
        betax = self.betx0
        betay = self.bety0
        alphax = self.alfx0
        alphay = self.alfy0
        Ex = self.emitta_x
        Ey = self.emitta_y
        nparticles = 5000
        # Set up beam tracking in MAD-X
        self.madx.use(SEQUENCE=self.sequence_name, range="#s/plasma_merge")
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=self.dx0, DPX=self.dpx0, BETY=self.bety0,
                                ALFY=self.alfy0, DY=self.dy0, DPY=self.dpy0, x=0, px=0, y=0, py=0)
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        for monitor in self.btv:
            self.madx.ptc_observe(place=monitor)
        ## Perform tracking
        with self.madx.batch():
            for i in range(nparticles):
                np.random.seed(seed + i)
                x0 = np.random.normal() * np.sqrt(Ex * betax) + 0.001
                np.random.seed(seed + 1 + i)
                y0 = np.random.normal() * np.sqrt(Ey * betay) - 0.001
                np.random.seed(seed + 2 + i)
                px0 = (np.random.normal() * np.sqrt(Ex * betax) - alphax * x0) / betax
                np.random.seed(seed + 3 + i)
                py0 = (np.random.normal() * np.sqrt(Ey * betay) - alphay * y0) / betay
                t0 = np.random.normal() * 0
                pt0 = np.random.normal() * self.mom_spr
                self.madx.ptc_start(x=x0, px=px0, y=y0, py=py0, t=t0, pt=pt0)
            self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                closed_orbit=False,
                                maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
            self.madx.ptc_track_end()
        ptc_output = self.madx.table.trackone
        return ptc_output

    def _pickle_data(self, estimates, data_all):
        # Save data in pickle format
        data_all['meas'] = estimates
        now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')
        if not self.bad_data:
            pickle.dump(data_all, open('n_screen_scan/data/data_n_screen_' + now + '.p', 'wb'))

    def _json_data(self, estimates, data_all):
        # Save data in JSON format
        data_all['meas'] = estimates
        now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')
        if not self.bad_data:
            with open('n_screen_scan/data/data_n_screen_' + now + '.json', 'w') as outfile:
                outfile.write(json.dumps(data_all, cls=myJSONEncoder))


