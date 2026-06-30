"""
run_cardiac.py
==============
Physiological cardiac-cycle simulation of a single conduit artery (aorta-like)
terminated by a three-element (R-C-R) Windkessel, driven by a realistic
ventricular-ejection inflow waveform. Two clinically-relevant parameter studies:

  (A) ARTERIAL STIFFENING / AGEING: sweep wall Young's modulus E so the
      foot-to-foot pulse-wave velocity (PWV) ranges from a compliant young
      artery (~5 m/s) to a stiff aged/hypertensive artery (~15 m/s). For each
      case we extract central blood pressure, augmentation index (AIx),
      pulse pressure (PP), and reflection-wave timing.

  (B) STENOSIS SEVERITY: introduce a localized luminal narrowing (area
      reduction 0-80%) and quantify the trans-stenotic pressure drop and the
      downstream flow waveform damping.

All outputs are real solver results written to ../results/*.csv and saved
waveforms to ../results/waveforms_*.npz for plotting.
"""
import os
import numpy as np
import csv
import sys
sys.path.insert(0, os.path.dirname(__file__))
import bloodflow1d as bf
from bloodflow1d import (Vessel1D, pressure, wave_speed, c0_speed,
                         beta_from_Eh, maccormack_step, cfl_dt, RHO)

RESDIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(RESDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

PA2MMHG = 1.0 / 133.322


# ---------------------------------------------------------------------------
# Inflow: physiological aortic-root flow (half-sinusoid systolic ejection).
# The model represents a SINGLE representative conduit artery, not the whole
# systemic circulation, so the inflow is the per-vessel volumetric flow whose
# amplitude is set by the characteristic impedance to give a physiological
# pulse pressure (~40 mmHg). HR 75 bpm -> 0.8 s cycle; 0.3 s ejection.
# ---------------------------------------------------------------------------
T_CYCLE = 0.8
T_EJECT = 0.3
Q_PEAK = 100e-6        # m^3/s (100 mL/s) per-vessel ejection peak


def inflow(t):
    tc = t % T_CYCLE
    if tc < T_EJECT:
        return Q_PEAK * np.sin(np.pi * tc / T_EJECT)
    return 0.0


def diastolic_pressure_init(A0, beta, p_d=10665.0):
    """Initial area giving a diastolic pressure p_d (default 80 mmHg)."""
    # p = beta/A0 (sqrt(A)-sqrt(A0))  -> solve for A
    sA = np.sqrt(A0) + p_d * A0 / beta
    return sA**2


def simulate(E, h=5e-4, r0=5e-3, L=0.4, N=401, n_cycles=6,
             Rp=None, Rd=None, C=1.0e-8, stenosis=None, record=True):
    """Run n_cycles cardiac cycles; return last-cycle waveforms at the proximal
    (inlet) and distal probes. stenosis = (center_frac, length_frac, area_frac)."""
    ves = Vessel1D(L, N, r0, E, h)
    A0_base = ves.A0
    beta = ves.beta

    # Spatially-varying A0 / beta for a stenosis (local lumen narrowing) plus a
    # Young--Tsai (1973) lumped pressure-loss term. A pure 1-D area reduction
    # alone under-predicts the trans-stenotic loss because the inviscid model
    # recovers most of the Bernoulli pressure downstream; the irreversible
    # viscous + separation (turbulent) losses are added through the standard
    # Young--Tsai stenosis-resistance source, distributed over the throat
    # segment. This is the established way to represent a stenosis in 1-D
    # arterial models (Young & Tsai 1973; Stergiopulos et al.).
    A0x = np.full(N, A0_base)
    sten_active = np.zeros(N, dtype=bool)
    Kv = 0.0
    Kt = 0.0
    Ls = 0.0
    area_ratio = 1.0
    if stenosis is not None:
        cfrac, lfrac, afrac = stenosis
        xc = cfrac * L
        half = 0.5 * lfrac * L
        mask = np.abs(ves.x - xc) < half
        prof = 0.5 * (1 + np.cos(np.pi * (ves.x - xc) / half))
        prof = np.clip(prof, 0, 1)
        A0x = A0_base * (1.0 - (1.0 - afrac) * np.where(mask, prof, 0.0))
        sten_active = mask
        Ls = 2.0 * half                    # stenosis length
        area_ratio = afrac                 # throat area / nominal area
        # Young--Tsai loss coefficients (Young & Tsai 1973):
        #   Kv (viscous) = 32 (Ls/D0) (A0/As),  Kt (turbulent) ~ 1.52
        D0 = 2.0 * r0
        Kv = 32.0 * (Ls / D0) * (1.0 / area_ratio)
        Kt = 1.52
    beta_x = beta_from_Eh(E, h, A0x)   # beta independent of A0 in this law

    # precompute per-node distribution weight of the lumped loss (so the
    # integral of the distributed sink over x equals the lumped Young--Tsai dp)
    n_sten = max(int(sten_active.sum()), 1)

    # Windkessel terminal resistance tuned so the mean transmural pressure is
    # ~93 mmHg with mean flow = stroke volume / cycle. The proximal resistance
    # Rp is matched to the vessel characteristic impedance Zc = rho c0 / A0 for
    # a (near-)reflection-free coupling; Rd + C set the diastolic decay.
    SV = np.trapezoid([inflow(t) for t in np.linspace(0, T_CYCLE, 400)],
                      np.linspace(0, T_CYCLE, 400))
    Qmean = SV / T_CYCLE
    p_mean_target = 93 * 133.322
    Rtot = p_mean_target / Qmean
    Zc = RHO * ves.c0 / A0_base
    if Rp is None:
        Rp = Zc                   # impedance-matched proximal resistance
    if Rd is None:
        Rd = max(Rtot - Rp, 0.1 * Rtot)

    # initialize the whole vessel at the diastolic operating pressure so the
    # recorded transmural pressure is the absolute arterial pressure.
    A_d = diastolic_pressure_init(A0_base, beta, p_d=80 * 133.322)
    ves.U[0, :] = A_d
    ves.U[1, :] = 0.0
    wk = {'pc': 93 * 133.322}

    dx = ves.dx
    x_prox = int(0.05 * N)
    x_mid = int(0.5 * N)
    x_dist = int(0.92 * N)
    # throat-adjacent probes: just upstream and just downstream of a mid-vessel
    # stenosis (segment half-width 0.075L -> 0.075*N nodes).
    half_nodes = int(0.075 * N) + 2
    x_up = max(1, x_mid - half_nodes)
    x_dn = min(N - 2, x_mid + half_nodes)

    # storage for last cycle
    last_t, last_p_prox, last_p_dist, last_q_prox, last_q_dist = [], [], [], [], []
    last_p_mid, last_q_mid = [], []
    last_p_up, last_p_dn = [], []
    t = 0.0
    t_end = n_cycles * T_CYCLE
    record_from = (n_cycles - 1) * T_CYCLE

    def stenosis_sink(A, Q):
        """Young--Tsai (1973) distributed momentum sink at the throat nodes.
        Lumped trans-stenotic loss
          dp = Kv * mu * Ubar / (A0)     +  Kt * (rho/2) (A0/As - 1)^2 Ubar|Ubar|
        is spread over the stenosis length Ls; the momentum-equation source is
          -A/rho * d(p_loss)/dx  ~  -A/rho * dp/Ls   at throat nodes."""
        s = np.zeros(N)
        if Kt == 0.0 and Kv == 0.0:
            return s
        # Canonical Young--Tsai (1973) lumped trans-stenotic pressure loss,
        # expressed in volumetric flow Q (single representative throat value):
        #   dp = Kv mu Q /(D0 A0) + Kt rho/(2 A0^2) (A0/As - 1)^2 Q|Q|
        D0 = 2.0 * r0
        Qs = Q[sten_active].mean()
        dp = (Kv * bf.MU * Qs / (D0 * A0_base)
              + Kt * RHO / (2.0 * A0_base ** 2)
              * (1.0 / area_ratio - 1.0) ** 2 * Qs * np.abs(Qs))
        # distribute the lumped loss uniformly over the throat nodes so that the
        # integral of the momentum sink reproduces the lumped pressure drop dp.
        s[sten_active] = -(A0_base / RHO) * (dp / Ls)
        return s

    # custom step using per-node A0x / beta_x
    def step(U, dt):
        A = U[0]; Q = U[1]
        # flux with spatially varying reference area
        B = beta_x / (3.0 * A0x * RHO) * A**1.5
        F = np.vstack([Q, Q * Q / A + B])
        S = np.vstack([np.zeros(N),
                       -8.0 * np.pi * bf.MU * Q / (RHO * A) + stenosis_sink(A, Q)])
        Up = U.copy()
        Up[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1]) + dt * S[:, :-1]
        Up[0] = np.maximum(Up[0], 1e-12)
        Ap = Up[0]; Qp = Up[1]
        Bp = beta_x / (3.0 * A0x * RHO) * Ap**1.5
        Fp = np.vstack([Qp, Qp * Qp / Ap + Bp])
        Sp = np.vstack([np.zeros(N),
                        -8.0 * np.pi * bf.MU * Qp / (RHO * Ap) + stenosis_sink(Ap, Qp)])
        Uc = U.copy()
        Uc[:, 1:] = 0.5 * (U[:, 1:] + Up[:, 1:]
                           - dt / dx * (Fp[:, 1:] - Fp[:, :-1]) + dt * Sp[:, 1:])
        Uc[0] = np.maximum(Uc[0], 1e-12)
        return Uc

    while t < t_end:
        dt = cfl_dt(ves.U, dx, A0x, beta_x.mean(), cfl=0.35)
        if t + dt > t_end:
            dt = t_end - t
        ves.U = step(ves.U, dt)
        # inlet: prescribe flow (use base A0/beta for inlet char.)
        ves.A0 = A0_base; ves.beta = beta
        ves.inlet_Q(inflow(t + dt))
        ves.outlet_windkessel(dt, Rp, Rd, C, wk)
        t += dt
        if record and t >= record_from:
            p = beta_x / A0x * (np.sqrt(ves.U[0]) - np.sqrt(A0x))
            last_t.append(t - record_from)
            last_p_prox.append(p[x_prox]); last_q_prox.append(ves.U[1, x_prox])
            last_p_mid.append(p[x_mid]);   last_q_mid.append(ves.U[1, x_mid])
            last_p_dist.append(p[x_dist]); last_q_dist.append(ves.U[1, x_dist])
            last_p_up.append(p[x_up]);     last_p_dn.append(p[x_dn])

    return dict(t=np.array(last_t),
                p_prox=np.array(last_p_prox), q_prox=np.array(last_q_prox),
                p_mid=np.array(last_p_mid), q_mid=np.array(last_q_mid),
                p_dist=np.array(last_p_dist), q_dist=np.array(last_q_dist),
                p_up=np.array(last_p_up), p_dn=np.array(last_p_dn),
                c0=ves.c0, Rp=Rp, Rd=Rd)


# ---------------------------------------------------------------------------
def measure_pwv(t, p_prox, p_dist, dist):
    """Foot-to-foot PWV. The foot of each pressure waveform is found by the
    intersecting-tangent method: the intersection of the horizontal line at the
    diastolic minimum with the tangent at the point of steepest systolic
    upstroke. PWV = separation / (t_foot_distal - t_foot_proximal)."""
    tu = np.linspace(t[0], t[-1], 1000)
    pp = np.interp(tu, t, p_prox)
    pd = np.interp(tu, t, p_dist)

    def foot_time(tt, p):
        pmin = p.min()
        dp = np.gradient(p, tt)
        i_up = int(np.argmax(dp))            # steepest upstroke
        slope = dp[i_up]
        if slope <= 0:
            return tt[0]
        # tangent: p = p[i_up] + slope*(t - tt[i_up]); intersect p = pmin
        t_foot = tt[i_up] + (pmin - p[i_up]) / slope
        return t_foot

    tf_p = foot_time(tu, pp)
    tf_d = foot_time(tu, pd)
    dt = tf_d - tf_p
    if dt <= 1e-4:
        return np.nan
    return dist / dt


def augmentation_index(t, p):
    """Augmentation index AIx = (P_sys - P1)/PP * 100, where P1 is the systolic
    shoulder (the inflection point that marks the arrival of the reflected wave)
    and P_sys the systolic peak. Standard tonometric definition (Nichols 2005).

    The shoulder is identified as the inflection (zero-crossing of the second
    derivative) on the systolic upstroke that lies closest before the systolic
    peak, restricted to the systolic ejection window to avoid foot/diastolic
    artefacts."""
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    # resample to a uniform, lightly-smoothed grid
    tu = np.linspace(t[0], t[-1], 600)
    pu = np.interp(tu, t, p)
    # light smoothing (moving average) to suppress numerical ripple in d2
    k = 7
    kern = np.ones(k) / k
    ps = np.convolve(pu, kern, mode='same')
    psys = pu.max()
    pdia = pu.min()
    PP = psys - pdia
    ipk = int(np.argmax(ps))
    dp = np.gradient(ps, tu)
    d2 = np.gradient(dp, tu)
    # candidate inflections strictly on the systolic upstroke (10%..peak),
    # and not within the first/last 3 samples
    lo = max(int(0.10 * ipk), 4)
    seg = d2[lo:ipk]
    if len(seg) > 2:
        zc = np.where(np.diff(np.sign(seg)))[0] + lo
    else:
        zc = np.array([], dtype=int)
    if len(zc):
        i1 = zc[-1]            # inflection nearest the peak = systolic shoulder
        P1 = ps[i1]
    else:
        # no distinct shoulder (very compliant case): take 70% up the upstroke
        i1 = max(1, int(0.7 * ipk))
        P1 = ps[i1]
    AIx = (psys - P1) / PP * 100.0
    return AIx, psys, pdia, PP, P1


# ---------------------------------------------------------------------------
def study_stiffness():
    print("\n=== Study A: arterial stiffening / ageing ===")
    # sweep E to span PWV ~5..15 m/s
    E_list = [3.0e5, 4.0e5, 6.0e5, 8.0e5, 1.1e6, 1.5e6]
    rows = [['E_Pa', 'c0_mps', 'PWV_mps', 'SBP_mmHg', 'DBP_mmHg',
             'PP_mmHg', 'AIx_percent']]
    waves = {}
    L = 0.4
    for E in E_list:
        s = simulate(E=E, L=L, N=401, n_cycles=6)
        t = s['t']
        # proximal and distal probes are at 0.05L and 0.92L -> separation 0.87L
        pwv = measure_pwv(t, s['p_prox'], s['p_dist'], 0.87 * L)
        AIx, psys, pdia, PP, P1 = augmentation_index(t, s['p_prox'])
        sbp = psys * PA2MMHG
        dbp = pdia * PA2MMHG
        ppm = PP * PA2MMHG
        print(f"E={E:.2e} c0={s['c0']:.2f} PWV={pwv:.2f} "
              f"SBP={sbp:.1f} DBP={dbp:.1f} PP={ppm:.1f} AIx={AIx:.1f}%")
        rows.append([f'{E:.3e}', f'{s["c0"]:.4f}', f'{pwv:.4f}',
                     f'{sbp:.2f}', f'{dbp:.2f}', f'{ppm:.2f}', f'{AIx:.2f}'])
        waves[f'E_{E:.2e}'] = dict(t=t, p_prox=s['p_prox'], q_prox=s['q_prox'],
                                   p_dist=s['p_dist'], c0=s['c0'], pwv=pwv,
                                   AIx=AIx)
    with open(os.path.join(RESDIR, 'study_stiffness.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    np.savez(os.path.join(RESDIR, 'waveforms_stiffness.npz'),
             **{k: v for kk, vv in waves.items()
                for k, v in {f'{kk}__{a}': b for a, b in vv.items()}.items()})
    return rows, waves


def study_stenosis():
    print("\n=== Study B: stenosis severity ===")
    sev_list = [0.0, 0.3, 0.5, 0.7, 0.8]   # area reduction fraction
    rows = [['area_reduction_pct', 'diam_reduction_pct',
             'dP_throat_mmHg', 'Q_distal_peak_mLs', 'Q_damping_pct']]
    waves = {}
    L = 0.4
    base = None
    dP_base = None
    for sev in sev_list:
        afrac = 1.0 - sev
        sten = (0.5, 0.15, afrac) if sev > 0 else None
        s = simulate(E=6.0e5, L=L, N=401, n_cycles=6, stenosis=sten)
        t = s['t']
        # trans-stenotic pressure drop = peak instantaneous (p_upstream - p_downstream)
        # measured immediately either side of the throat, with the unobstructed
        # (0 %) segment value subtracted to isolate the stenotic contribution.
        dP_raw = (s['p_up'] - s['p_dn']).max() * PA2MMHG
        if dP_base is None:
            dP_base = dP_raw
        dP = dP_raw - dP_base
        qd_peak = s['q_dist'].max() * 1e6
        if base is None:
            base = qd_peak
        damp = (1 - qd_peak / base) * 100 if base else 0.0
        diam_red = (1 - np.sqrt(afrac)) * 100
        print(f"area_red={sev*100:.0f}% (diam {diam_red:.0f}%) "
              f"dP={dP:.2f} mmHg Qd_peak={qd_peak:.1f} mL/s damping={damp:.1f}%")
        rows.append([f'{sev*100:.0f}', f'{diam_red:.1f}', f'{dP:.3f}',
                     f'{qd_peak:.2f}', f'{damp:.2f}'])
        waves[f'sev_{sev:.2f}'] = dict(t=t, p_prox=s['p_prox'],
                                       p_dist=s['p_dist'], q_dist=s['q_dist'])
    with open(os.path.join(RESDIR, 'study_stenosis.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    np.savez(os.path.join(RESDIR, 'waveforms_stenosis.npz'),
             **{k: v for kk, vv in waves.items()
                for k, v in {f'{kk}__{a}': b for a, b in vv.items()}.items()})
    return rows, waves


if __name__ == '__main__':
    study_stiffness()
    study_stenosis()
    print("\nDone. CSVs + waveforms in results/.")
