# %% ----- imports
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import clipboard  # for clipboard
import collections
from scipy.constants import c

import pynlo
from edf.re_nlse_joint_5level import EDF
import blit
from edf import edfa
from edf.utility import crossSection, ER80_4_125_betas

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
cm = 1e-2
km = 1e3

W = 1.0


output = collections.namedtuple("output", ["model", "sim"])
n_records = 100


def propagate(fiber, pulse, length):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse, t_shock=None, raman_on=False)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- load dispersion coefficients ------------------------------
polyfit = ER80_4_125_betas().polyfit

D_g_a = 18
polyfit_a = np.array([-(1550e-9**2) / (2 * np.pi * c) * (D_g_a * ps / nm / km)])

# %% ------------- pulse ------------------------------------------------------
f_r = 200e6
n = 256
v_min = c / 1650e-9
v_max = c / 1500e-9
v0 = c / 1560e-9
e_p = 1e-3 / f_r

t_fwhm = 2e-12
min_time_window = 20e-12
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
    alias=2,
)
dv_dl = pulse.v_grid**2 / c

# %% --------- passive fibers -------------------------------------------------
gamma_pm1550 = 1.2
gamma_edf = 1.2

pm1550 = pynlo.materials.SilicaFiber()
# pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.set_beta_from_beta_n(v0, polyfit_a)  # only gdd
pm1550.gamma = gamma_pm1550 / (W * km)

# %% ------------ active fiber ------------------------------------------------
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_a)  # only gdd
# edf.load_fiber_from_dict(pynlo.materials.pm1550)  # set edf dispersion same as pm1550
edf.gamma = gamma_edf / (W * km)

# %% ------- linear cavity ------
l_p = 0.3
l_g = 0.16
Pp = 100 * 1e-3

alpha_ns = 0.05
delta_T = 0.1
I_sat = 9 * 1e-6 / cm**2 * (np.pi * 5e-6**2)
T = lambda e_p: 1 - alpha_ns - delta_T / (1 + e_p / I_sat)

pulse.a_t[:] = np.random.uniform(0, 1, size=pulse.n)
pulse.e_p = 0.1e-6 / f_r  # .1 microwats
p_fwd = pulse.copy()
p_bck = pulse.copy()
p_out = pulse.copy()

fig, ax = plt.subplots(1, 2)

done = False
loop_count = 0
while not done:
    p_fwd.a_t[:] = propagate(pm1550, p_fwd, l_p).sim.pulse_out.a_t[:]

    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=p_fwd,
        p_bck=p_bck,
        edf=edf,
        length=l_g,
        Pp_fwd=Pp,
        Pp_bck=0.0,
        t_shock=None,
        raman_on=False,
        n_records=n_records,
    )
    p_fwd.a_t[:] = sim_fwd.pulse_out.a_t[:]
    p_bck.a_t[:] = sim_bck.pulse_out.a_t[:]

    T_curve = T(p_fwd.p_t * pulse.dt)
    p_fwd.p_t[:] *= T_curve

    p_bck.a_t[:] = propagate(pm1550, p_bck, l_p).sim.pulse_out.a_t[:]
    p_bck.a_t[:] *= 0.8**0.5
    p_bck.a_t[:] = propagate(pm1550, p_bck, l_p).sim.pulse_out.a_t[:]

    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=p_bck,
        p_bck=p_fwd,
        edf=edf,
        length=l_g,
        Pp_fwd=Pp,
        Pp_bck=0.0,
        t_shock=None,
        raman_on=False,
        n_records=n_records,
    )
    p_bck.a_t[:] = sim_fwd.pulse_out.a_t[:]
    p_fwd.a_t[:] = sim_bck.pulse_out.a_t[:]

    T_curve = T(p_bck.p_t * pulse.dt)
    p_bck.p_t[:] *= T_curve

    p_fwd.a_t[:] = propagate(pm1550, p_fwd, l_p).sim.pulse_out.a_t[:]
    p_out.a_t[:] = p_fwd.a_t[:] * 0.2**0.5
    p_fwd.a_t[:] *= 0.8**0.5

    center = pulse.n // 2
    p_fwd.p_t[:] = np.roll(p_fwd.p_t, center - p_fwd.p_t.argmax())
    p_bck.p_t[:] = np.roll(p_bck.p_t, center - p_bck.p_t.argmax())

    if loop_count == 0:
        (l1,) = ax[0].plot(
            p_out.wl_grid * 1e9,
            p_out.p_v / p_out.p_v.max() * dv_dl,
            animated=True,
        )
        (l2,) = ax[1].plot(
            p_out.t_grid * 1e12,
            p_out.p_t / p_out.p_t.max(),
            animated=True,
        )
        (l3,) = ax[1].plot(
            p_out.t_grid * 1e12,
            T_curve,
            animated=True,
        )
        ax[1].set_ylim(ymin=0)

        fr_number = ax[0].annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )
        fig.tight_layout()
        bm = blit.BlitManager(fig.canvas, [l1, l2, l3, fr_number])
        bm.update()

    else:
        l1.set_ydata(p_out.p_v / p_out.p_v.max() * dv_dl)
        l2.set_ydata(p_out.p_t / p_out.p_t.max())
        l3.set_ydata(T_curve)
        fr_number.set_text(f"loop #: {loop_count}")
        bm.update()

    loop_count += 1

    print(loop_count, p_out.e_p * f_r * 1e3)
