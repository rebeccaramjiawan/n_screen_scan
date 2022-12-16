import seaborn as sns
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

params = {'axes.labelsize': 24,  # fontsize for x and y labels (was 10)
          'axes.titlesize': 24,
          'legend.fontsize': 20,  # was 10
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.linewidth': 1.5,
          'lines.linewidth': 3,
          'text.usetex': True,
          'font.family': 'serif'
          }
plt.rcParams.update(params)


def gaussian(x, amp, cen, wid, off):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + off


def plot_btv(btv_calc_inst, btv, btvs, gamma_r, estimates, data_xy):
    # Plot BTV data with estimates for beam parameters

    btv_in = btv
    epsilon_x = estimates["epsilon_x"]
    epsilon_y = estimates["epsilon_y"]

    for i in range(len(btv)):
        btv = btv_in
        if i == 0:
            beta_x = beta_x_0 = estimates["beta_x"]
            beta_y = beta_y_0 = estimates["beta_y"]
            alpha_x = alpha_x_0 = estimates["alpha_x"]
            alpha_y = alpha_y_0 = estimates["alpha_y"]
            gamma_x_0 = estimates["gamma_x"]
            gamma_y_0 = estimates["gamma_y"]
            disp_x = estimates["dispersion_x"]
            disp_y = estimates["dispersion_y"]
            deriv_disp_x = estimates["deriv_dispersion_x"]
            deriv_disp_y = estimates["deriv_dispersion_y"]
            delta = estimates["delta_y"]
        else:
            # R** parameters are matrix parameters quantifying 6D beam propagation
            r11, r12, r33, r34, r21, r22, r43, r44 = btv_calc_inst.r_val_calc(btv[0], btv[i])
            beta_x = beta_x_0 * r11 ** 2 - 2 * r11 * r12 * alpha_x_0 + gamma_x_0 * r12 ** 2
            beta_y = beta_y_0 * r33 ** 2 - 2 * r33 * r34 * alpha_y_0 + gamma_y_0 * r34 ** 2
            alpha_x = -beta_x_0 * r11 * r21 + r22 * r11 * alpha_x_0 + r12 * r21 * alpha_x_0 - gamma_x_0 * r12 * r22
            alpha_y = -beta_y_0 * r11 * r21 + r22 * r11 * alpha_y_0 + r12 * r21 * alpha_y_0 - gamma_y_0 * r12 * r22
        table_values = [["$\epsilon_x$", np.round(gamma_r * epsilon_x, 3), "$\mu$m"],
                        ["$\epsilon_y$", np.round(gamma_r * epsilon_y, 3), "$\mu$m"],
                        [r"$\beta_x$", np.round(beta_x, 3), "m"], [r"$\beta_y$", np.round(beta_y, 3), "m"],
                        [r"$\alpha_x$", np.round(alpha_x, 3), ""], [r"$\alpha_y$", np.round(alpha_y, 3), ""]]

        # The higher the dimensionality of the input data the more parameters you can estimate
        if len(btv) > 3:
            table_values = np.vstack(
                (table_values, [r"$D_x$", np.round(disp_x, 3), "m"], [r"$D_y$", np.round(disp_y, 3), "m"]))
        if len(btv) > 4:
            table_values = np.vstack((table_values, [r"$D'_x$", np.round(deriv_disp_x, 3), "m"],
                                      [r"$D'_y$", np.round(deriv_disp_y, 3), "m"]))
        if len(btv) > 5:
            table_values = np.vstack((table_values, [r"$\delta$", np.round(delta * 100, 4), "\%"]))

        btv = btvs
        xmin = data_xy[btv[i]]["x"][0]
        xmax = data_xy[btv[i]]["x"][-1]
        ymin = data_xy[btv[i]]["y"][0]
        ymax = data_xy[btv[i]]["y"][-1]
        xmin_int = int(np.ceil(xmin))
        ymin_int = int(np.ceil(ymin))
        xmax_int = int(np.floor(xmax))
        ymax_int = int(np.floor(ymax))

        x_axis_labels = np.linspace(xmin_int, xmax_int, (xmax_int - xmin_int) + 1)
        y_axis_labels = np.linspace(ymin_int, ymax_int, (ymax_int - ymin_int) + 1)

        fig, ax = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [5, 1, 1], 'height_ratios': [1, 4]})
        plt.suptitle("%s" % btv[i], fontsize=30)

        g = sns.heatmap(np.transpose(data_xy[btv[i]]["image"]), xticklabels=x_axis_labels, yticklabels=y_axis_labels,
                        cbar=False, cmap="mako",
                        ax=ax[1][0])

        x_lim = ax[1][0].get_xlim()[1]
        y_lim = ax[1][0].get_ylim()[0]
        #
        ax[1][0].set_yticks(
            np.arange(y_lim * (ymin_int - ymin) / (ymax - ymin), y_lim + 1 - y_lim * (ymax - ymax_int) / (ymax - ymin),
                      step=y_lim / (ymax - ymin)))
        ax[1][0].set_xticks(
            np.arange(x_lim * (xmin_int - xmin) / (xmax - xmin), x_lim + 1 - x_lim * (xmax - xmax_int) / (xmax - xmin),
                      step=x_lim / (xmax - xmin)))
        g.set_xticklabels(labels=x_axis_labels, rotation=0)
        g.set_yticklabels(labels=y_axis_labels, rotation=0)
        ax[1][0].box = "on"
        ax[1][0].invert_yaxis()
        ax[1][0].set_xlabel("$x$ [mm]")
        ax[1][0].set_ylabel("$y$ [mm]")
        #
        ax[0][0].get_xaxis().set_visible(False)
        ax[0][0].get_yaxis().set_visible(False)
        for pos in ['top', 'left', 'right']:
            ax[0][0].spines[pos].set_visible(False)

        x_ax = data_xy[btv[i]]["x"]
        y_ax = data_xy[btv[i]]["y"]

        ax[0][0].plot(x_ax, data_xy[btv[i]]["pro_x"], color="darkblue")
        ax[0][0].plot(x_ax, data_xy[btv[i]]["fit_x"], color="darkturquoise")
        try:
            ax[0][0].text(0.75, 0.15, "Mean = %.1f$\pm$%.1f mm \n R.M.S = %.1f$\pm$%.1f mm \n Amp. %s a.u" % (
                (data_xy[btv[i]]["mu_x"]), (data_xy[btv[i]]["err_x"]), data_xy[btv[i]]["sigma_x"],
                (data_xy[btv[i]]["mu_err_x"]), data_xy[btv[i]]["amp"]),
                          fontsize=20, transform=ax[0][0].transAxes)
        except:
            pass

        ax[1][1].get_xaxis().set_visible(False)
        ax[1][1].get_yaxis().set_visible(False)
        for pos in ['top', 'bottom', 'right']:
            ax[1][1].spines[pos].set_visible(False)
        ax[1][1].plot(data_xy[btv[i]]["pro_y"], y_ax, color="darkblue")
        ax[1][1].plot(data_xy[btv[i]]["fit_y"], y_ax, color="darkturquoise")
        try:
            ax[1][1].text(-0.05, -0.2,
                          "Mean = %.1f $\pm$ %.1f mm \n R.M.S = %.1f $\pm$ %.1f mm" % (
                              (data_xy[btv[i]]["mu_y"]),(data_xy[btv[i]]["err_y"]),
                              data_xy[btv[i]]["sigma_y"],
                              (data_xy[btv[i]]["mu_err_y"])), fontsize=20,
                          transform=ax[1][1].transAxes)
        except:
            pass

        table = ax[1][2].table(cellText=table_values, colLabels=[r"$\bf{Param.}$", r"$\bf{Value}$", r"$\bf{Units}$"],
                               edges="open", loc='upper center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(2.2, 2.5)

        ax[0][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        plt.subplots_adjust(left=0.08, bottom=0.2, top=0.9, right=0.92)
        plt.show()


def plot_twiss(btv_calc_inst, btv, btvs, gamma_r, estimates, data_xy, twiss):
    # Propagate Twiss values calculated from data using the beamline model

    plt.figure(figsize=(15, 8))
    plt.plot(twiss["s"], twiss["betx"], '--', color="tab:blue", label=r"design $\beta_x$")
    plt.plot(twiss["s"], twiss["bety"], '--', color="darkorange", label=r"design $\beta_y$")
    for i in twiss["name"]:
        if twiss["s"][twiss["name"] == i] < twiss["s"][twiss["name"] == (btv[0] + ":1")]:
            pass
        elif i[:-2] == btv[0]:
            beta_x = beta_x_0 = estimates["beta_x"]
            beta_y = beta_y_0 = estimates["beta_y"]
            alpha_x = alpha_x_0 = estimates["alpha_x"]
            alpha_y = alpha_y_0 = estimates["alpha_y"]
            gamma_x_0 = estimates["gamma_x"]
            gamma_y_0 = estimates["gamma_y"]
            disp_x = estimates["dispersion_x"]
            disp_y = estimates["dispersion_y"]
            deriv_disp_x = estimates["deriv_dispersion_x"]
            deriv_disp_y = estimates["deriv_dispersion_y"]
            delta = estimates["delta_y"]
            s = twiss["s"][twiss["name"] == (btv[0] + ":1")][0]
        elif "drift" in i:
            pass
        else:
            r11, r12, r33, r34, r21, r22, r43, r44 = btv_calc_inst.r_val_calc(btv[0], i[:-2])
            beta_x = np.append(beta_x, beta_x_0 * r11 ** 2 - 2 * r11 * r12 * alpha_x_0 + gamma_x_0 * r12 ** 2)
            beta_y = np.append(beta_y, beta_y_0 * r33 ** 2 - 2 * r33 * r34 * alpha_y_0 + gamma_y_0 * r34 ** 2)
            alpha_x = -beta_x_0 * r11 * r21 + r22 * r11 * alpha_x_0 + r12 * r21 * alpha_x_0 - gamma_x_0 * r12 * r22
            alpha_y = -beta_y_0 * r11 * r21 + r22 * r11 * alpha_y_0 + r12 * r21 * alpha_y_0 - gamma_y_0 * r12 * r22
            s = np.append(s, twiss["s"][twiss["name"] == i])

    plt.plot(s, beta_x, color="tab:blue", label=r"propagated $\beta_x$")
    plt.plot(s, beta_y, color="darkorange", label=r"propagated $\beta_y$")
    plt.xlabel("$s$ [m]", labelpad=0.5)
    plt.title(r"$\beta$ functions propagated from initial estimate at BTV")
    plt.ylabel(r"$\beta_x, \beta_y$ [m]", labelpad=5)
    plt.ylim(-20, np.max(np.hstack((twiss["betx"], twiss["bety"], beta_x, beta_y))) * 1.1)
    plt.xlim(0, twiss["s"][-1])
    for i in btv:
        plt.plot([twiss["s"][twiss["name"] == (i + ":1")][0], twiss["s"][twiss["name"] == (i + ":1")][0]], [-100, 400],
                 '--',
                 color="gray", linewidth=1)
    plt.legend()
