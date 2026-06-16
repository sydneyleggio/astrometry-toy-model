import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# ============================================================
#                     CONSTANTS 
# ============================================================
sigma_rad     = (1  * u.mas).to(u.rad).value
dt_seconds    = (30 * u.min).to(u.s).value
T_obs_seconds = (3.5 * u.yr).to(u.s).value

f_l   = 1.0 / T_obs_seconds          # low-frequency cutoff  (= 1/T_obs)
f_yr  = (1 / u.yr).to(u.Hz).value    # 1/year in Hz
A_gw  = 1e-15                         # NANOGrav GW amplitude
alpha = -2.0 / 3.0                    # spectral index

ELL_MIN = 2
ELL_MAX = 100     # truncation order; increase for finer convergence
N_THETA = 1000    # number of Theta sample points in [0, pi]


# ============================================================
#  SPECTRAL QUANTITIES 
# ============================================================

def P_gw(f):
    """GW power spectrum."""
    return (A_gw**2 / (12.0 * np.pi**2)) * (f / f_yr)**(2 * alpha) * f**(-1)


def S_h(f):
    """One-sided strain PSD:  S_h = 12 pi^2 P_gw(f)."""
    return 12.0 * np.pi**2 * P_gw(f)


# ============================================================
#  MODE-COUPLING COEFFICIENT
# ============================================================

def F_sq(ell):
    """
    |F_ell^E|^2 = |F_ell^B|^2 = 1 / ( N_ell^2 * ell*(ell+1) )
    N_ell^2 = (ell+2)(ell+1)ell(ell-1) / 2
    Cross terms vanish.
    """
    N_sq = (ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0) / 2.0
    return 1.0 / (N_sq * ell * (ell + 1.0))


# ============================================================
#  ANGULAR POWER SPECTRA
# ============================================================

def C_EE(ell, f):
    """C_ell^EE = 16 pi |F_ell^E|^2 S_h(f)."""
    return 16.0 * np.pi * F_sq(ell) * S_h(f)


def C_BB(ell, f):
    """C_ell^BB = 16 pi |F_ell^B|^2 S_h(f).
    For an isotropic background C_BB = C_EE."""
    return 16.0 * np.pi * F_sq(ell) * S_h(f)


# ============================================================
#  VECTORISED LEGENDRE RECURRENCE 
# ============================================================

def compute_legendre_recurrence(mu, ell_max):
    """
    Returns P_l^0, P_l^1, P_l^2  for l = 0 … ell_max.
    Arrays have shape (ell_max+1, *mu.shape).
    """
    shape = mu.shape
    P0 = np.zeros((ell_max + 1,) + shape)
    P1 = np.zeros((ell_max + 1,) + shape)
    P2 = np.zeros((ell_max + 1,) + shape)

    sin_t = np.sqrt(np.maximum(1.0 - mu**2, 0.0))

    P0[0] = 1.0
    P0[1] = mu
    P1[1] = -sin_t
    P2[2] = 3.0 * sin_t**2          # P_2^2 = 3 sin^2(theta)

    for l in range(1, ell_max):
        P0[l + 1] = ((2*l + 1) * mu * P0[l] - l * P0[l - 1]) / (l + 1)
        if l >= 1:
            P1[l + 1] = (
                ((2*l + 1) * mu * P1[l] - (l + 1) * P1[l - 1]) / l
                if l >= 2 else (2*l + 1) * mu * P1[l]
            )
        if l >= 2:
            P2[l + 1] = (
                ((2*l + 1) * mu * P2[l] - (l + 2) * P2[l - 1]) / (l - 1)
                if l >= 3 else (2*l + 1) * mu * P2[l]
            )

    return P0, P1, P2


# ============================================================
#  G KERNELS
# ============================================================

def G1(ell, P0_ell, P2_ell):
    """G_l^(1)(Theta) = -1/2 [ P_l^2(cos Theta)/(l(l+1)) - P_l(cos Theta) ]"""
    ll1 = ell * (ell + 1.0)
    return -0.5 * (P2_ell / ll1 - P0_ell)


def G2(ell, P1_ell, theta):
    """G_l^(2)(Theta) = -1/(l(l+1)) * P_l^1(cos Theta)/sin(Theta)

    Singularities resolved analytically:
        Theta -> 0 :  lim = +0.5  (for all ell >= 2)
        Theta -> pi:  lim =  0.0
    """
    ll1   = ell * (ell + 1.0)
    sin_t = np.sin(theta)

    mask_zero = np.abs(theta)          < 1e-12
    mask_pi   = np.abs(theta - np.pi) < 1e-12
    mask_reg  = ~mask_zero & ~mask_pi

    g2 = np.zeros_like(theta)
    g2[mask_reg]  = -P1_ell[mask_reg] / (ll1 * sin_t[mask_reg])
    g2[mask_zero] = 0.5
    g2[mask_pi]   = 0.0
    return g2


# ============================================================
#  C_parallel(Theta)
# ============================================================

def C_parallel(theta_arr, f, ell_min=ELL_MIN, ell_max=ELL_MAX):
    """
    C^parallel(Theta) = sum_{ell=ell_min}^{ell_max}
        (2*ell + 1) / (4*pi)  *  [ C_ell^EE * G1_ell(Theta)
                                  + C_ell^BB * G2_ell(Theta) ]

    Singularities at Theta = 0 and Theta = pi are handled inside G2.
    """
    theta_arr = np.asarray(theta_arr, dtype=float)
    mu = np.cos(theta_arr)

    P0, P1, P2 = compute_legendre_recurrence(mu, ell_max)

    total = np.zeros_like(theta_arr)
    for ell in range(ell_min, ell_max + 1):
        g1  = G1(ell, P0[ell], P2[ell])
        g2  = G2(ell, P1[ell], theta_arr)
        Cee = C_EE(ell, f)
        Cbb = C_BB(ell, f)
        total += (2.0 * ell + 1.0) / (4.0 * np.pi) * (Cee * g1 + Cbb * g2)

    # Enforce the analytic limit at Theta = pi explicitly
    total[np.abs(theta_arr - np.pi) < 1e-12] = 0.0

    return total


# ============================================================
#  MAIN — evaluate and plot
# ============================================================

def main():
    # Dense grid over [0, pi]
    theta = np.linspace(0.0, np.pi, N_THETA)

    C = C_parallel(theta, f_l)

    print(f'  S_h(f_l)           = {S_h(f_l):.4e}')
    print(f'  C^parallel (0)     = {C[0]:.4e}')
    print(f'  C^parallel (pi)    = {C[-1]:.4e}')

    # ── Plot ────────────────────────────────────────────────────────────────
    theta_deg = np.rad2deg(theta)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(theta_deg, C, color='steelblue', linewidth=2,
            label=r'$C^{\parallel}(\Theta)$')
    ax.axhline(0.0, color='grey', linewidth=0.8, linestyle='--')

    ax.set_xlabel(r'$\Theta$ (degrees)', fontsize=13)
    ax.set_ylabel(r'$C^{\parallel}(\Theta)$', fontsize=13)
    ax.set_title(
        r'Astrometric Correlation Function $C^{\parallel}(\Theta)$',
        fontsize=12
    )
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()