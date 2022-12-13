import numpy as np
import plotting as plot
import BTV_calculations as btv_calc

import configparser

config = configparser.RawConfigParser()
config.read('n_screen_scan/BTV_config.txt')

btvs = config['INPUT']["btvs"].split(",")
btv_madx = config['INPUT']["btv_madx"].split(",")
btv_screens = config['INPUT']["btv_screens"].split(",")
particle = config['INPUT']["particle"]  # e (electrons) or p (protons)
mass = float(config['INPUT']["mass"])
energy = float(config['INPUT']["energy"])
n_shot = int(config['INPUT']["n_shot"])
n_btv_x = int(config['INPUT']["n_btv_x"])
n_btv_y = int(config['INPUT']["n_btv_y"])
simulate = config['INPUT']["simulate"] == "True"  # if true requires emitta, mom_spr to be set
is_rms = config['INPUT']["is_rms"] == "True"  # False gives you standard deviation
emitta_x = float(config['INPUT']["emitta_x"])
emitta_y = float(config['INPUT']["emitta_y"])
mom_spr = float(config['INPUT']["mom_spr"])
# emitta = []
# mom_spr = []
gamma_r = np.sqrt(energy ** 2 + mass ** 2) / mass

if (n_btv_x > len(btv_madx)) or (n_btv_x > len(btv_madx)):
    print("** Number of BTVs requested exceeds number of BTVs provided **")
if len(btvs) != len(btv_madx):
    print("** Number of BTVs does not equal number of screens **")

btv_madx = [x.lower() for x in btv_madx]

btv_calc_inst = btv_calc.BTV_calc(energy, mass, emitta_x, emitta_y, mom_spr, btv_madx, btvs, btv_screens, simulate, particle,
                                  n_btv_x, n_btv_y)
estimates, data_xy, twiss = btv_calc_inst.emit_calc(n_shot, is_rms)

plot.plot_btv(btv_calc_inst, btv_madx, btvs, gamma_r, estimates, data_xy)
plot.plot_twiss(btv_calc_inst, btv_madx, btvs, gamma_r, estimates, data_xy, twiss)
