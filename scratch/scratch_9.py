"""
also using the amplify function from scratch_7, this time setting a nonzero
gamma, and stretching in dcf beforehand

"""
# %% ----- imports
import sys

sys.path.append("../")
from scipy.constants import c
import clipboard
from scratch_6 import EDF
import pynlo
import numpy as np
import matplotlib.pyplot as plt
import scratch_7 as edfa
import collections
from edf.utility import crossSection


ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])


def propagate(fiber, pulse, length, n_records=None):
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
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# %% -------------- load absorption coefficients from NLight ------------------
spl_sigma_a = crossSection().sigma_a
spl_sigma_e = crossSection().sigma_e

# %% -------------- load dispersion coefficients ------------------------------
gamma_a = 1 / (W * km)

# %% ------------- pulse ------------------------------------------------------
f_r = 100e6
e_p = 100e-3 / f_r

n = 256
v_min = c / 1850e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
t_fwhm = 50e-15
min_time_window = 250e-12
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
dv_dl = pulse.v_grid**2 / c  # J / Hz -> J / m

# %% ------------ chirp fiber ------------------------------------------------
dcf = pynlo.materials.SilicaFiber()
D_g_2 = -100
polyfit = np.array([-(1550e-9**2) / (2 * np.pi * c) * (D_g_2 * ps / nm / km)])
dcf.set_beta_from_beta_n(v0, polyfit)
dcf.gamma = 0

length_dcf = 10.0
model_dcf, sim_dcf = propagate(dcf, pulse, length_dcf)
pulse_dcf = sim_dcf.pulse_out

# %% ------------ active fiber ------------------------------------------------
n_ion = 55 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm
r_eff = 5.5 * um
a_eff = np.pi * r_eff**2
n_ion_Y = n_ion * 10

overlap_p = 5.5**2 / 65**2
overlap_s = 1.0

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

length = 4.5

edf = EDF(
    f_r=f_r,
    overlap_p=overlap_p,
    overlap_s=1.0,
    n_ion=n_ion,
    n_ion_Y=n_ion_Y,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
D_g_2 = 18
polyfit = np.array([-(1550e-9**2) / (2 * np.pi * c) * (D_g_2 * ps / nm / km)])
edf.set_beta_from_beta_n(v0, polyfit)
beta_n = edf._beta(pulse.v_grid)
edf.gamma = gamma_a

# %% ----------- edfa ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=pulse_dcf,
    p_bck=None,
    edf=edf,
    length=length,
    Pp_fwd=18,
    Pp_bck=0,
    n_records=100,
    raman_on=True,
)
sim = sim_fwd

# %% ----------- plot results -------------------------------------------------
sol_Pp = sim.Pp
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n
na = sim.na_n
nb = sim.nb_n

print("output power is: ", sol_Ps[-1])

fig = plt.figure(
    figsize=np.array([11.16, 5.21]),
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(z, sol_Pp, label="pump", linewidth=2)
ax1.plot(z, sol_Ps, label="signal", linewidth=2)
ax1.grid()
ax1.legend(loc="upper left")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

ax2.plot(z, n1, label="n1", linewidth=2)
ax2.plot(z, n2, label="n2", linewidth=2)
ax2.plot(z, n3, label="n3", linewidth=2)
ax2.plot(z, n4, label="n4", linewidth=2)
ax2.plot(z, n5, label="n5", linewidth=2)
ax2.plot(z, na, label="na", linewidth=2)
ax2.plot(z, nb, label="nb", linewidth=2)
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")
fig.tight_layout()

sim.plot("wvl")

fig, ax = plt.subplots(1, 2)
p_wl = sim.p_v * dv_dl
ax[0].plot(pulse.wl_grid * 1e9, p_wl[0] / p_wl[0].max())
ax[0].plot(pulse.wl_grid * 1e9, p_wl[-1] / p_wl[-1].max())
ax[1].plot(pulse.t_grid * 1e12, sim.p_t[0] / sim.p_t[0].max())
ax[1].plot(pulse.t_grid * 1e12, sim.p_t[-1] / sim.p_t[-1].max())
ax[0].set_xlabel("wavelength (nm)")
ax[1].set_xlabel("time (ps)")
fig.tight_layout()
