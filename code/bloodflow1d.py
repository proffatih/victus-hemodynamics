"""
bloodflow1d.py
==============
One-dimensional (1-D) arterial blood-flow solver for the reduced
Navier-Stokes equations (conservation of mass + momentum + elastic tube law),
solved with a second-order MacCormack finite-difference scheme.

Governing equations (area-flow A-Q form), e.g. Sherwin et al. (2003),
Olufsen et al. (2000), Alastruey et al. (2011):

    dA/dt + dQ/dx = 0
    dQ/dt + d/dx( alpha * Q^2 / A + (1/rho) * integral( A dp ) ) = - 2 (gamma+2) pi mu Q / (rho A)

with the elastic (thin-wall, linearly-elastic) tube law

    p(A) = p_ext + beta/A0 * ( sqrt(A) - sqrt(A0) ),   beta = (4/3) sqrt(pi) E h

and pulse wave speed  c = sqrt( beta / (2 rho A0) ) * A^{1/4}.

We write the system in conservative form U=(A,Q), F=(Q, alpha Q^2/A + B(A)),
S=(0, viscous + dB/dA0 elastic-taper terms). For a uniform vessel the elastic
pressure flux integral B(A) has the closed form used below.

Author: Fatih Gül (auto-generated solver, no fabricated data).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI units). Realistic blood / arterial values from
# literature (Olufsen 2000; Sherwin 2003; Alastruey 2011; Reymond 2009).
# ---------------------------------------------------------------------------
RHO = 1060.0          # blood density [kg/m^3]
MU = 4.0e-3           # dynamic viscosity [Pa.s]
ALPHA = 1.0           # momentum-flux (Coriolis) correction, flat profile
GAMMA = 2.0           # velocity-profile parameter (=2 -> 2(gamma+2)=8 -> Poiseuille)
P_EXT = 0.0           # external/reference pressure [Pa]


def beta_from_Eh(E, h, A0):
    """Stiffness coefficient beta = (4/3) sqrt(pi) E h  [Pa.m]."""
    return (4.0 / 3.0) * np.sqrt(np.pi) * E * h


def pressure(A, A0, beta):
    """Elastic tube law p(A) [Pa]."""
    return P_EXT + beta / A0 * (np.sqrt(A) - np.sqrt(A0))


def wave_speed(A, A0, beta):
    """Local pulse wave speed c(A) [m/s]."""
    return np.sqrt(beta / (2.0 * RHO * A0)) * A**0.25


def c0_speed(A0, beta):
    """Reference (diastolic) Moens-Korteweg-type wave speed at A=A0."""
    return np.sqrt(beta / (2.0 * RHO * A0)) * A0**0.25


def flux(U, A0, beta):
    """Conservative flux F(U). U shape (2, N)."""
    A = U[0]
    Q = U[1]
    F0 = Q
    # B(A) = beta/(3 A0 rho) * A^{3/2}  is the integral term giving the
    # pressure gradient contribution; d/dx of (alpha Q^2/A + B) is the flux.
    B = beta / (3.0 * A0 * RHO) * A**1.5
    F1 = ALPHA * Q * Q / A + B
    return np.vstack([F0, F1])


def source(U):
    """Source S(U): viscous wall friction term."""
    A = U[0]
    Q = U[1]
    S0 = np.zeros_like(A)
    S1 = -2.0 * (GAMMA + 2.0) * np.pi * MU * Q / (RHO * A)
    return np.vstack([S0, S1])


def maccormack_step(U, dt, dx, A0, beta):
    """One MacCormack predictor-corrector step (interior nodes)."""
    F = flux(U, A0, beta)
    S = source(U)
    # Predictor (forward differences)
    Up = U.copy()
    Up[:, :-1] = U[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1]) + dt * S[:, :-1]
    # enforce positivity for tube law sqrt
    Up[0] = np.maximum(Up[0], 1e-12)
    Fp = flux(Up, A0, beta)
    Sp = source(Up)
    # Corrector (backward differences)
    Uc = U.copy()
    Uc[:, 1:] = 0.5 * (U[:, 1:] + Up[:, 1:]
                       - dt / dx * (Fp[:, 1:] - Fp[:, :-1])
                       + dt * Sp[:, 1:])
    Uc[0] = np.maximum(Uc[0], 1e-12)
    return Uc


def cfl_dt(U, dx, A0, beta, cfl=0.4):
    A = U[0]
    Q = U[1]
    u = Q / A
    c = wave_speed(A, A0, beta)
    smax = np.max(np.abs(u) + c)
    return cfl * dx / smax


class Vessel1D:
    """Single uniform elastic vessel, A-Q form, MacCormack solver."""

    def __init__(self, L, N, r0, E, h):
        self.L = L
        self.N = N
        self.x = np.linspace(0.0, L, N)
        self.dx = self.x[1] - self.x[0]
        self.r0 = r0
        self.A0 = np.pi * r0**2
        self.E = E
        self.h = h
        self.beta = beta_from_Eh(E, h, self.A0)
        self.c0 = c0_speed(self.A0, self.beta)
        # initial state: at rest, A=A0, Q=0
        self.U = np.vstack([np.full(N, self.A0), np.zeros(N)])

    # --- boundary conditions via characteristics (Riemann invariants) ---
    def _W_forward(self, A, Q):
        """Forward Riemann invariant W1 = u + 4 c."""
        u = Q / A
        c = wave_speed(A, self.A0, self.beta)
        return u + 4.0 * c

    def _W_backward(self, A, Q):
        u = Q / A
        c = wave_speed(A, self.A0, self.beta)
        return u - 4.0 * c

    def inlet_Q(self, Qin):
        """Prescribe inlet flow Qin using outgoing W2 from interior."""
        A1, Q1 = self.U[0, 1], self.U[1, 1]
        W2 = self._W_backward(A1, Q1)         # outgoing (leftward) invariant
        # Solve for A at inlet given Q=Qin and W2 = Qin/A - 4 c(A)
        cc = np.sqrt(self.beta / (2.0 * RHO * self.A0))
        # f(A) = Qin/A - 4 cc A^{1/4} - W2 = 0
        A = self.A0
        for _ in range(60):
            c = cc * A**0.25
            f = Qin / A - 4.0 * c - W2
            df = -Qin / A**2 - cc * A**(-0.75)
            A = A - f / df
            A = max(A, 1e-10)
        self.U[0, 0] = A
        self.U[1, 0] = Qin

    def outlet_windkessel(self, dt, Rp, Rd, C, p_out_state):
        """3-element Windkessel outlet (R-C-R). Returns updated reservoir p."""
        A_n, Q_n = self.U[0, -2], self.U[1, -2]
        W1 = self._W_forward(A_n, Q_n)       # outgoing (rightward) invariant
        pc = p_out_state['pc']               # capacitor pressure
        # Outflow Q satisfies: p(A) = pc + Rp*Q ; and W1 = Q/A + 4 c(A).
        cc = np.sqrt(self.beta / (2.0 * RHO * self.A0))
        A = self.U[0, -1]
        for _ in range(60):
            c = cc * A**0.25
            p = pressure(A, self.A0, self.beta)
            Q = (p - pc) / Rp
            f = Q / A + 4.0 * c - W1
            # derivatives wrt A
            dp = 0.5 * self.beta / self.A0 / np.sqrt(A)
            dQ = dp / Rp
            df = dQ / A - Q / A**2 + cc * A**(-0.75)
            A = A - f / df
            A = max(A, 1e-10)
        p = pressure(A, self.A0, self.beta)
        Q = (p - pc) / Rp
        self.U[0, -1] = A
        self.U[1, -1] = Q
        # update capacitor pressure: C dpc/dt = Q - (pc - p_venous)/Rd
        p_ven = 0.0
        dpc = dt * (Q - (pc - p_ven) / Rd) / C
        p_out_state['pc'] = pc + dpc
        return p_out_state

    def outlet_reflection(self, Rt):
        """Pure reflection-coefficient outlet (Rt in [-1,1]).
        Rt=0 -> non-reflecting; Rt=1 -> closed (full positive reflection)."""
        A_n, Q_n = self.U[0, -2], self.U[1, -2]
        W1 = self._W_forward(A_n, Q_n)
        # initial outgoing characteristic value relative to rest
        W1_0 = 4.0 * self.c0
        dW1 = W1 - W1_0
        dW2 = -Rt * dW1
        W2 = -4.0 * self.c0 + dW2
        # solve A,Q from W1,W2: u=(W1+W2)/2 ; c=(W1-W2)/8
        u = 0.5 * (W1 + W2)
        c = (W1 - W2) / 8.0
        cc = np.sqrt(self.beta / (2.0 * RHO * self.A0))
        A = (c / cc)**4
        Q = u * A
        self.U[0, -1] = A
        self.U[1, -1] = Q
