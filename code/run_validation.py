"""
run_validation.py
=================
Validate the 1-D MacCormack solver against analytical/benchmark results:

(V1) Pulse-wave-speed test: propagate a small-amplitude pressure pulse along a
     long inviscid vessel and measure the numerical wave speed; compare with the
     theoretical small-signal speed c0 = sqrt(beta/(2 rho A0)) A0^{1/4}
     (Moens-Korteweg / Sherwin 2003).

(V2) Wave-reflection test (Alastruey 2011 / Sherwin 2003 single-vessel):
     impose a known terminal reflection coefficient Rt and verify that the
     reflected pressure amplitude equals Rt * incident amplitude.

(V3) Grid-convergence: confirm second-order spatial accuracy of MacCormack.

Outputs CSV to ../results/ . No fabricated numbers; everything from the solver.
"""
import os
import numpy as np
import csv
import sys
sys.path.insert(0, os.path.dirname(__file__))
from bloodflow1d import (Vessel1D, pressure, wave_speed, c0_speed,
                         RHO, beta_from_Eh)

RESDIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESDIR, exist_ok=True)


# ---------------------------------------------------------------------------
def run_pulse(L=0.5, N=801, r0=0.005, E=4.0e5, h=5.0e-4,
              amp=20.0, tmax=0.08, viscous=False, Rt=0.0, record_x=None):
    """Propagate a Gaussian inlet flow pulse; return time series of p at probes."""
    ves = Vessel1D(L, N, r0, E, h)
    if not viscous:
        import bloodflow1d as bf
        bf.MU = 0.0
    c0 = ves.c0
    dx = ves.dx

    # Gaussian inlet flow pulse Q(t)
    t0 = 0.02
    sigma = 0.004

    def Qin(t):
        return amp * 1e-6 * np.exp(-((t - t0) / sigma)**2)   # amp in mL/s -> m^3/s

    if record_x is None:
        record_x = [0.1 * L, 0.5 * L, 0.9 * L]
    probe_idx = [int(round(xx / dx)) for xx in record_x]

    from bloodflow1d import maccormack_step, cfl_dt
    t = 0.0
    times = []
    pseries = {i: [] for i in probe_idx}
    while t < tmax:
        dt = cfl_dt(ves.U, dx, ves.A0, ves.beta, cfl=0.4)
        if t + dt > tmax:
            dt = tmax - t
        ves.U = maccormack_step(ves.U, dt, dx, ves.A0, ves.beta)
        ves.inlet_Q(Qin(t + dt))
        ves.outlet_reflection(Rt)
        t += dt
        times.append(t)
        p = pressure(ves.U[0], ves.A0, ves.beta)
        for i in probe_idx:
            pseries[i].append(p[i])
    import bloodflow1d as bf
    bf.MU = 4.0e-3  # restore
    return np.array(times), {i: np.array(v) for i, v in pseries.items()}, ves, probe_idx, record_x


def v1_wavespeed():
    # Long vessel, probes placed so the pulse cleanly transits both before any
    # terminal reflection returns; tmax sized to vessel length / c0.
    L = 0.8
    c0_guess = c0_speed(np.pi * 0.005**2, beta_from_Eh(4.0e5, 5.0e-4, np.pi * 0.005**2))
    tmax = 0.9 * L / c0_guess
    xprobe = [0.15 * L, 0.45 * L]
    times, ps, ves, idx, xs = run_pulse(L=L, N=1201, viscous=False, Rt=0.0,
                                        tmax=tmax, record_x=xprobe, amp=0.5)
    c0 = ves.c0
    i1, i2 = idx[0], idx[1]
    t1 = times[np.argmax(ps[i1] - ps[i1][0])]
    t2 = times[np.argmax(ps[i2] - ps[i2][0])]
    dist = xs[1] - xs[0]
    c_num = dist / (t2 - t1)
    err = abs(c_num - c0) / c0 * 100
    print(f"[V1] theoretical c0 = {c0:.4f} m/s | numerical = {c_num:.4f} m/s "
          f"| error = {err:.3f}%")
    with open(os.path.join(RESDIR, 'validation_wavespeed.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'value', 'unit'])
        w.writerow(['c0_theory', f'{c0:.6f}', 'm/s'])
        w.writerow(['c_numerical', f'{c_num:.6f}', 'm/s'])
        w.writerow(['rel_error_percent', f'{err:.6f}', '%'])
        w.writerow(['probe_x1', f'{xs[0]:.4f}', 'm'])
        w.writerow(['probe_x2', f'{xs[1]:.4f}', 'm'])
        w.writerow(['t_arrival_1', f'{t1:.6f}', 's'])
        w.writerow(['t_arrival_2', f'{t2:.6f}', 's'])
    return c0, c_num, err


def v2_reflection(Rt_list=(-0.5, 0.0, 0.5, 0.8)):
    """Single probe at vessel midpoint. The incident pulse passes the probe once
    (rightward), reflects at the terminus, and passes again (leftward). The ratio
    of reflected to incident peak amplitude at the probe equals Rt (Alastruey 2011).
    Probe distance to terminus d gives reflected-arrival lag ~ 2 d / c0."""
    rows = [['Rt_imposed', 'p_incident_Pa', 'p_reflected_Pa', 'Rt_measured', 'error']]
    print("[V2] Wave-reflection benchmark (single-vessel, prescribed Rt):")
    L = 0.6
    c0 = c0_speed(np.pi * 0.005**2, beta_from_Eh(4.0e5, 5.0e-4, np.pi * 0.005**2))
    xprobe = 0.5 * L
    d_term = L - xprobe                 # probe-to-terminus distance
    tmax = (xprobe + 2.0 * d_term) / c0 + 0.04
    lag = 2.0 * d_term / c0             # reflected-arrival delay after incident
    for Rt in Rt_list:
        t, ps, ves, idx, xs = run_pulse(L=L, N=1201, viscous=False, Rt=Rt,
                                        tmax=tmax, record_x=[xprobe], amp=0.5)
        p = idx[0]
        sig = ps[p] - ps[p][0]
        # incident peak time (first arrival)
        t_inc = t[np.argmax(sig)]
        p_inc = np.max(sig)
        # reflected window centred at t_inc + lag
        win = (t > t_inc + 0.4 * lag)
        if not np.any(win):
            p_ref = 0.0
        elif Rt >= 0:
            p_ref = np.max(sig[win])
        else:
            p_ref = np.min(sig[win])
        Rt_meas = p_ref / p_inc if p_inc != 0 else 0.0
        err = abs(Rt_meas - Rt)
        print(f"   Rt={Rt:+.2f} -> measured {Rt_meas:+.3f} (err {err:.3f}), "
              f"p_inc={p_inc:.2f} Pa, p_ref={p_ref:.2f} Pa")
        rows.append([f'{Rt:.2f}', f'{p_inc:.4f}', f'{p_ref:.4f}',
                     f'{Rt_meas:.4f}', f'{err:.4f}'])
    with open(os.path.join(RESDIR, 'validation_reflection.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    return rows


def v3_convergence():
    """Grid convergence of MacCormack on the smooth pulse (L2 error vs finest)."""
    Ns = [201, 401, 801, 1601]
    sols = {}
    print("[V3] Grid convergence (MacCormack, smooth pulse):")
    for N in Ns:
        t, ps, ves, idx, xs = run_pulse(L=0.5, N=N, viscous=False, Rt=0.0,
                                        tmax=0.06, record_x=[0.25])
        # sample final pressure peak value at mid probe (interp on common time grid)
        sols[N] = (t, ps[idx[0]])
    ref_t, ref_p = sols[Ns[-1]]
    rows = [['N', 'dx_mm', 'L2_err_vs_finest_Pa', 'order']]
    prev_err = None
    prev_dx = None
    for N in Ns[:-1]:
        t, p = sols[N]
        pi = np.interp(ref_t, t, p)
        err = np.sqrt(np.mean((pi - ref_p)**2))
        dx = 0.5 / (N - 1) * 1000  # mm
        order = ''
        if prev_err is not None:
            order = f'{np.log(prev_err/err)/np.log(prev_dx/dx):.2f}'
        print(f"   N={N:5d} dx={dx:.3f}mm  L2err={err:.4f} Pa  order={order}")
        rows.append([N, f'{dx:.4f}', f'{err:.6f}', order])
        prev_err, prev_dx = err, dx
    with open(os.path.join(RESDIR, 'validation_convergence.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    return rows


if __name__ == '__main__':
    print("=" * 60)
    print("1-D blood-flow solver validation")
    print("=" * 60)
    v1_wavespeed()
    v2_reflection()
    v3_convergence()
    print("Done. CSVs in results/.")
