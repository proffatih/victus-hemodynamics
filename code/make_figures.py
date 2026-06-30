"""
make_figures.py
===============
Generate publication-quality figures (vector PDF + 300 dpi PNG, colorblind-safe)
from the solver results. All numbers are read from ../results/.
"""
import os
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sys
sys.path.insert(0, os.path.dirname(__file__))
import bloodflow1d as bf
from bloodflow1d import Vessel1D, pressure, c0_speed, beta_from_Eh

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG, exist_ok=True)
PA2MMHG = 1.0 / 133.322

# Okabe-Ito colorblind-safe palette
CB = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.9, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.8, 'mathtext.fontset': 'cm',
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300)
    plt.close(fig)
    print('  wrote', name)


def read_csv(path):
    with open(path) as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------------------
def fig1_schematic():
    """Vessel + Windkessel + stenosis schematic."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis('off')
    # vessel walls
    x = np.linspace(0, 8, 400)
    r = np.full_like(x, 0.55)
    # stenosis bump
    sten = 0.35 * np.exp(-((x - 5.0) / 0.45)**2)
    rup = r - sten
    ax.fill_between(x, rup, 1.1, color='#c9d9e8', zorder=0)
    ax.fill_between(x, -rup, -1.1, color='#c9d9e8', zorder=0)
    ax.plot(x, rup, color=CB[0], lw=2)
    ax.plot(x, -rup, color=CB[0], lw=2)
    ax.fill_between(x, -rup, rup, color='#e8453c', alpha=0.18, zorder=1)
    # inflow arrow + label
    ax.annotate('', xy=(0.9, 0), xytext=(-0.4, 0),
                arrowprops=dict(arrowstyle='-|>', color=CB[1], lw=2.5))
    ax.text(-0.5, 0.78, r'$Q_{in}(t)$ (LV ejection)', color=CB[1], fontsize=11)
    # x axis
    ax.annotate('', xy=(8.2, -1.35), xytext=(0, -1.35),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
    ax.text(8.0, -1.65, r'$x$', fontsize=12)
    ax.text(2.3, 1.30, r'elastic artery: $p(A)=\dfrac{\beta}{A_0}(\sqrt{A}-\sqrt{A_0})$',
            fontsize=10, ha='center')
    # stenosis label
    ax.annotate('stenosis\n(local $A_0$ reduction)', xy=(5.0, 0.22), xytext=(6.6, 1.55),
                ha='center', fontsize=9, color=CB[1],
                arrowprops=dict(arrowstyle='->', color=CB[1]))
    # Windkessel box at outlet
    bx, by = 8.4, 0
    ax.add_patch(plt.Rectangle((bx, by - 0.75), 1.5, 1.5, fill=False,
                               edgecolor='k', lw=1.2))
    ax.text(bx + 0.75, by + 0.95, '3-element\nWindkessel', ha='center', fontsize=8.5)
    ax.text(bx + 0.75, by, r'$R_p\!-\!C\!-\!R_d$', ha='center', fontsize=9)
    ax.annotate('', xy=(bx, 0), xytext=(8.05, 0),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
    ax.set_xlim(-0.7, 10.2); ax.set_ylim(-1.9, 1.9)
    ax.set_aspect('auto')
    save(fig, 'fig1_schematic')


def fig2_waveforms():
    """Pressure & flow waveforms along the vessel for a representative case."""
    d = np.load(os.path.join(RES, 'waveforms_stiffness.npz'))
    # pick the most compliant (young) baseline key
    keys = sorted({k.split('__')[0] for k in d.files})
    base = keys[1]  # second-lowest E -> healthy adult
    t = d[f'{base}__t']
    pp = d[f'{base}__p_prox'] * PA2MMHG
    pd = d[f'{base}__p_dist'] * PA2MMHG
    qp = d[f'{base}__q_prox'] * 1e6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.plot(t, pp, color=CB[0], label='proximal (root)')
    ax1.plot(t, pd, color=CB[1], ls='--', label='distal')
    ax1.set_xlabel('time (s)'); ax1.set_ylabel('pressure (mmHg)')
    ax1.set_title('(a) Pressure waveforms'); ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)
    ax2.plot(t, qp, color=CB[2])
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('flow rate (mL s$^{-1}$)')
    ax2.set_title('(b) Proximal flow waveform'); ax2.grid(alpha=0.3)
    save(fig, 'fig2_waveforms')


def fig3_validation():
    """Validation overlay: wave-speed error + reflection benchmark + convergence."""
    ws = read_csv(os.path.join(RES, 'validation_wavespeed.csv'))
    wsd = {r[0]: r[1] for r in ws[1:]}
    refl = read_csv(os.path.join(RES, 'validation_reflection.csv'))[1:]
    conv = read_csv(os.path.join(RES, 'validation_convergence.csv'))[1:]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.2))
    # panel a: wave speed bar
    c_th = float(wsd['c0_theory']); c_num = float(wsd['c_numerical'])
    err = float(wsd['rel_error_percent'])
    ax1.bar([0, 1], [c_th, c_num], color=[CB[0], CB[1]], width=0.55)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(['theory\n(Moens-Korteweg)', 'numerical'])
    ax1.set_ylabel('pulse-wave speed (m s$^{-1}$)')
    ax1.set_title(f'(a) Wave speed (err {err:.2f}%)')
    ax1.set_ylim(0, c_th * 1.3)
    # panel b: reflection
    Rt_imp = [float(r[0]) for r in refl]
    Rt_meas = [float(r[3]) for r in refl]
    ax2.plot([-1, 1], [-1, 1], 'k--', lw=1, label='ideal $R_t$')
    ax2.scatter(Rt_imp, Rt_meas, color=CB[2], s=55, zorder=3, label='numerical')
    ax2.set_xlabel('imposed reflection coeff. $R_t$')
    ax2.set_ylabel('measured $R_t$')
    ax2.set_title('(b) Wave reflection'); ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)
    # panel c: convergence
    dx = [float(r[1]) for r in conv]
    err2 = [float(r[2]) for r in conv]
    ax3.loglog(dx, err2, 'o-', color=CB[3], label='MacCormack')
    ref2 = np.array(err2[0]) * (np.array(dx) / dx[0])**2
    ax3.loglog(dx, ref2, 'k:', lw=1.2, label='2nd-order slope')
    ax3.set_xlabel('grid spacing $\\Delta x$ (mm)')
    ax3.set_ylabel('$L_2$ error (Pa)')
    ax3.set_title('(c) Grid convergence'); ax3.legend(frameon=False)
    ax3.grid(alpha=0.3, which='both')
    save(fig, 'fig3_validation')


def fig4_stiffness_waveforms():
    """Overlaid proximal pressure waveforms for increasing stiffness (ageing)."""
    d = np.load(os.path.join(RES, 'waveforms_stiffness.npz'))
    keys = sorted({k.split('__')[0] for k in d.files},
                  key=lambda s: float(s.split('_')[1]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
    for c, k in zip(cmap, keys):
        t = d[f'{k}__t']; p = d[f'{k}__p_prox'] * PA2MMHG
        pwv = float(d[f'{k}__pwv'])
        ax.plot(t, p, color=c, label=f'PWV={pwv:.1f} m/s')
    ax.set_xlabel('time (s)'); ax.set_ylabel('central pressure (mmHg)')
    ax.set_title('Central pressure vs arterial stiffness')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, 'fig4_stiffness_waveforms')


def fig5_aix_pwv():
    """Headline clinical plot: AIx and PP vs PWV (arterial ageing)."""
    rows = read_csv(os.path.join(RES, 'study_stiffness.csv'))[1:]
    pwv = [float(r[2]) for r in rows]
    aix = [float(r[6]) for r in rows]
    pp = [float(r[5]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    l1, = ax1.plot(pwv, aix, 'o-', color=CB[0], label='Augmentation index')
    ax1.set_xlabel('pulse-wave velocity, PWV (m s$^{-1}$)')
    ax1.set_ylabel('Augmentation index, AIx (%)', color=CB[0])
    ax1.tick_params(axis='y', labelcolor=CB[0])
    ax2 = ax1.twinx()
    l2, = ax2.plot(pwv, pp, 's--', color=CB[1], label='Pulse pressure')
    ax2.set_ylabel('pulse pressure, PP (mmHg)', color=CB[1])
    ax2.tick_params(axis='y', labelcolor=CB[1])
    ax1.grid(alpha=0.3)
    ax1.legend(handles=[l1, l2], frameon=False, loc='upper left')
    ax1.set_title('Wave reflection increases with arterial stiffening')
    save(fig, 'fig5_aix_pwv')


def fig6_stenosis():
    """Stenosis severity: pressure drop + distal flow damping."""
    rows = read_csv(os.path.join(RES, 'study_stenosis.csv'))[1:]
    sev = [float(r[0]) for r in rows]
    dP = [float(r[2]) for r in rows]
    damp = [float(r[4]) for r in rows]
    d = np.load(os.path.join(RES, 'waveforms_stenosis.npz'))
    keys = sorted({k.split('__')[0] for k in d.files},
                  key=lambda s: float(s.split('_')[1]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    fig.subplots_adjust(wspace=0.55)
    ax1.plot(sev, dP, 'o-', color=CB[1], label='trans-stenotic $\\Delta p$')
    ax1.set_xlabel('area reduction (%)')
    ax1.set_ylabel('peak-systolic $\\Delta p$ (mmHg)', color=CB[1])
    ax1.tick_params(axis='y', labelcolor=CB[1])
    axb = ax1.twinx()
    axb.plot(sev, damp, 's--', color=CB[0])
    axb.set_ylabel('distal flow damping (%)', color=CB[0])
    axb.tick_params(axis='y', labelcolor=CB[0])
    ax1.set_title('(a) Stenosis severity effect'); ax1.grid(alpha=0.3)
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(keys)))
    for c, k in zip(cmap, keys):
        t = d[f'{k}__t']; q = d[f'{k}__q_dist'] * 1e6
        lvl = float(k.split('_')[1]) * 100
        ax2.plot(t, q, color=c, label=f'{lvl:.0f}% red.')
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('distal flow (mL s$^{-1}$)')
    ax2.set_title('(b) Distal flow waveforms'); ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)
    save(fig, 'fig6_stenosis')


def fig7_spacetime():
    """Space-time pressure map showing pulse propagation + reflection, using the
    same calibrated parameters as the cardiac study (physiological pressures)."""
    import bloodflow1d as bfm
    from bloodflow1d import maccormack_step, cfl_dt, RHO
    L = 0.4; N = 401
    ves = Vessel1D(L, N, 5e-3, 6e5, 5e-4)
    A0 = ves.A0; beta = ves.beta
    Tc = 0.8; Te = 0.3; Qpk = 100e-6
    # diastolic init at 80 mmHg (matches simulate())
    A_d = (np.sqrt(A0) + 80*133.322*A0/beta)**2
    ves.U[0, :] = A_d; ves.U[1, :] = 0.0

    def qin(tt):
        tcc = tt % Tc
        return Qpk*np.sin(np.pi*tcc/Te) if tcc < Te else 0.0

    SV = Qpk*Te*2/np.pi
    Rtot = (93*133.322)/(SV/Tc)
    Zc = RHO*ves.c0/A0
    Rp = Zc; Rd = max(Rtot-Rp, 0.1*Rtot)
    wk = {'pc': 93*133.322}
    n_settle = 2
    t = 0.0
    T = (n_settle+1)*Tc
    record_from = n_settle*Tc
    snaps = []; ts = []
    while t < T:
        dt = cfl_dt(ves.U, ves.dx, A0, beta, 0.35)
        ves.U = maccormack_step(ves.U, dt, ves.dx, A0, beta)
        ves.inlet_Q(qin(t+dt))
        ves.outlet_windkessel(dt, Rp, Rd, 1e-8, wk)
        t += dt
        if t >= record_from and (len(ts) == 0 or t-ts[-1] > 5e-3):
            # subsample the spatial grid to keep the raster light
            snaps.append((pressure(ves.U[0], A0, beta)*PA2MMHG)[::4])
            ts.append(t-record_from)
    snaps = np.array(snaps)
    xg = ves.x[::4]*100
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    im = ax.pcolormesh(xg, np.array(ts), snaps, shading='auto',
                       cmap='RdBu_r', rasterized=True)
    ax.set_xlabel('axial position $x$ (cm)'); ax.set_ylabel('time (s)')
    ax.set_title('Pulse-wave propagation and reflection')
    cb = fig.colorbar(im, ax=ax); cb.set_label('pressure (mmHg)')
    save(fig, 'fig7_spacetime')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_schematic()
    fig3_validation()
    fig2_waveforms()
    fig4_stiffness_waveforms()
    fig5_aix_pwv()
    fig6_stenosis()
    fig7_spacetime()
    print('All figures written to figures/.')
