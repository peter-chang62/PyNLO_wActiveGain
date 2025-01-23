# -----------------------------------------------------------------------------
# The rate equations for Ytterbium coupling are not analytic; the cross terms
# mean they are no longer linear!

# -----------------------------------------------------------------------------

# %% ----- imports
import numpy as np
from scipy.constants import h

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3

nm = 1e-9
um = 1e-6
cm = 1e-2
km = 1e3
W = 1.0

# Q. Han, J. Ning, and Z. Sheng, "Numerical Investigation of the ASE and Power
# Scaling of Cladding-Pumped Er–Yb Codoped Fiber Amplifiers," IEEE J. Quantum
# Electron. 46, 1535–1541 (2010).

# lifetimes
tau_21 = 10 * ms
tau_32 = 1.0 * ns  # much faster than Barmenkov et al.
tau_ba = 1.5 * ms

# cross-relaxation between Yb <-> Er
R = 2.371e-22  # m^3/s
sigma_a_p_Y = 1.2e-20 * cm**2
sigma_e_p_Y = 1.2e-20 * cm**2
Cup = 3e-24  # m^3/s


def dn1_dt(
    n1,
    n2,
    n3,
    na,
    nb,
    Ner,
    Nyb,
    sigma_12,
    sigma_21,
    sigma_13,
    tau_21,
    Cup,
    R,
):
    return (
        -sigma_12 * n1
        + sigma_21 * n2
        - sigma_13 * n1
        + n2 / tau_21
        + Cup * n2**2 * Ner
        - R * nb * Nyb * n1
        + R * na * Nyb * n3
    )


def dn2_dt(
    n1,
    n2,
    n3,
    Ner,
    sigma_12,
    sigma_21,
    tau_21,
    tau_32,
    Cup,
):
    return (
        sigma_12 * n1
        - sigma_21 * n2
        - n2 / tau_21
        - 2 * Cup * n2**2 * Ner
        + n3 / tau_32
    )


def dn3_dt(
    n1,
    n2,
    n3,
    na,
    nb,
    Ner,
    Nyb,
    sigma_13,
    tau_32,
    Cup,
    R,
):
    return (
        sigma_13 * n1
        - n3 / tau_32
        + Cup * n2**2 * Ner
        + R * nb * Nyb * n1
        - R * na * Nyb * n3
    )


def dna_dt(
    na,
    nb,
    n1,
    n3,
    Ner,
    sigma_ab,
    sigma_ba,
    tau_ba,
    R,
):
    return (
        -sigma_ab * na
        + sigma_ba * nb
        + nb / tau_ba
        + R * nb * n1 * Ner
        - R * na * n3 * Ner
    )


def dnb_dt(
    na,
    nb,
    n1,
    n3,
    Ner,
    sigma_ab,
    sigma_ba,
    tau_ba,
    R,
):
    return (
        sigma_ab * na
        - sigma_ba * nb
        - nb / tau_ba
        - R * nb * n1 * Ner
        + R * na * n3 * Ner
    )


def _factor_sigma(sigma, nu, P, overlap, A):
    return overlap * sigma * P / (h * nu * A)


def dn1_dt_func(
    a_eff,
    overlap_p,
    overlap_s,
    nu_p,
    P_p,
    nu_s,
    P_s,
    n1,
    n2,
    n3,
    na,
    nb,
    Ner,
    Nyb,
    sigma_a,
    sigma_e,
    sigma_p,
    tau_21,
    Cup,
    R,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)

    return dn1_dt(
        n1,
        n2,
        n3,
        na,
        nb,
        Ner,
        Nyb,
        sigma_12,
        sigma_21,
        sigma_13,
        tau_21,
        Cup,
        R,
    )


def dn2_dt_func(
    a_eff,
    overlap_s,
    nu_s,
    P_s,
    n1,
    n2,
    n3,
    Ner,
    sigma_a,
    sigma_e,
    tau_21,
    tau_32,
    Cup,
):
    sigma_12 = _factor_sigma(sigma_a, nu_s, P_s, overlap_s, a_eff)
    sigma_21 = _factor_sigma(sigma_e, nu_s, P_s, overlap_s, a_eff)
    if isinstance(P_s, np.ndarray) and P_s.size > 1:
        sigma_12 = np.sum(sigma_12)
        sigma_21 = np.sum(sigma_21)

    return dn2_dt(
        n1,
        n2,
        n3,
        Ner,
        sigma_12,
        sigma_21,
        tau_21,
        tau_32,
        Cup,
    )


def dn3_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    n1,
    n2,
    n3,
    na,
    nb,
    Ner,
    Nyb,
    sigma_p,
    tau_32,
    Cup,
    R,
):
    sigma_13 = _factor_sigma(sigma_p, nu_p, P_p, overlap_p, a_eff)

    return dn3_dt(
        n1,
        n2,
        n3,
        na,
        nb,
        Ner,
        Nyb,
        sigma_13,
        tau_32,
        Cup,
        R,
    )


def dna_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    na,
    nb,
    n1,
    n3,
    Ner,
    sigma_a_p_Y,
    sigma_e_p_Y,
    tau_ba,
    R,
):
    sigma_ab = _factor_sigma(sigma_a_p_Y, nu_p, P_p, overlap_p, a_eff)
    sigma_ba = _factor_sigma(sigma_e_p_Y, nu_p, P_p, overlap_p, a_eff)

    return dna_dt(
        na,
        nb,
        n1,
        n3,
        Ner,
        sigma_ab,
        sigma_ba,
        tau_ba,
        R,
    )


def dnb_dt_func(
    a_eff,
    overlap_p,
    nu_p,
    P_p,
    na,
    nb,
    n1,
    n3,
    Ner,
    sigma_a_p_Y,
    sigma_e_p_Y,
    tau_ba,
    R,
):
    sigma_ab = _factor_sigma(sigma_a_p_Y, nu_p, P_p, overlap_p, a_eff)
    sigma_ba = _factor_sigma(sigma_e_p_Y, nu_p, P_p, overlap_p, a_eff)

    return dnb_dt(
        na,
        nb,
        n1,
        n3,
        Ner,
        sigma_ab,
        sigma_ba,
        tau_ba,
        R,
    )


def dPp_dz(
    overlap_p,
    P_p,
    n1,
    na,
    nb,
    Ner,
    Nyb,
    sigma_p,
    sigma_a_p_Y,
    sigma_e_p_Y,
):
    n1 *= Ner
    na *= Nyb
    nb *= Nyb
    return (-sigma_p * n1 - sigma_a_p_Y * na + sigma_e_p_Y * nb) * overlap_p * P_p


def dPs_dz(
    overlap_s,
    P_s,
    n1,
    n2,
    Ner,
    sigma_a,
    sigma_e,
):
    n1 *= Ner
    n2 *= Ner
    return (-sigma_a * n1 + sigma_e * n2) * overlap_s * P_s


def gain(
    overlap_s,
    n1,
    n2,
    Ner,
    sigma_a,
    sigma_e,
):
    n1 *= Ner
    n2 *= Ner
    return (-sigma_a * n1 + sigma_e * n2) * overlap_s
