import numpy as np
import n_screen_scan.plotting as plot
import n_screen_scan.BTV_calculations as btv_calc
import configparser


def main():
    # Inputs sources from BTV_config.txt file
    config = configparser.RawConfigParser()
    config.read('n_screen_scan/BTV_config.txt')

    btvs = config['INPUT']["btvs"].split(",")
    btv_madx = config['INPUT']["btv_madx"].split(",")
    btv_screens = config['INPUT']["btv_screens"].split(",")
    particle = config['INPUT']["particle"]  # e (electrons) or p (protons)
    mass = float(config['INPUT']["mass"])
    energy = float(config['INPUT']["energy"])  # beam energy in MeV
    n_shot = int(config['INPUT']["n_shot"])  # number of shots to average
    n_btv_x = int(config['INPUT']["n_btv_x"])  # number of horizontal BTVs to use
    n_btv_y = int(config['INPUT']["n_btv_y"])  # number of vertical BTVs to use
    simulate = config['INPUT']["simulate"] == "True"  # if true then requires emitta and mom_spr to be set
    is_rms = config['INPUT']["is_rms"] == "True"  # False gives you standard deviation
    emitta_x = float(config['INPUT']["emitta_x"])  # Horizontal emittance
    emitta_y = float(config['INPUT']["emitta_y"])  # Vertical emittnce
    mom_spr = float(config['INPUT']["mom_spr"])  # Momentum spread
    gamma_r = np.sqrt(energy ** 2 + mass ** 2) / mass  # Lorentz gamma

    # Check number of BTVs match that requested
    if (n_btv_x > len(btv_madx)) or (n_btv_x > len(btv_madx)):
        print("** Number of BTVs requested exceeds number of BTVs provided **")
    if len(btvs) != len(btv_madx):
        print("** Number of BTVs does not equal number of screens **")

    btv_madx = [x.lower() for x in btv_madx]

    btv_calc_inst = btv_calc.BTV_calc(energy, mass, emitta_x, emitta_y, mom_spr, btv_madx, btvs, btv_screens, simulate,
                                      particle,
                                      n_btv_x, n_btv_y)  # Calculation  methods in BTV_calc
    estimates, data_xy, twiss = btv_calc_inst.emit_calc(n_shot, is_rms)

    plot.plot_btv(btv_calc_inst, btv_madx, btvs, gamma_r, estimates,
                  data_xy)  # plot data from BTV with estimates for beam parameters
    plot.plot_twiss(btv_calc_inst, btv_madx, btvs, gamma_r, estimates, data_xy, twiss)  # propagate Twiss parameters


if __name__ == "__main__":
    main()
