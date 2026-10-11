# Physics of the P00 forward model

All formulas use SI units, peak phasors and the conventions in
`CONVENTIONS.md`. Sources: Morozovska, Eliseev and Kalinin, J. Appl. Phys. 111,
014107 (2012), doi:10.1063/1.3675508; Kalinin et al., doi:10.1063/1.2206992;
Eliseev et al., arXiv:1403.1225; the P00 Patch 3 and Patch 4 notes of
`2026_ESM_oracle` for the derived coefficients.

Symbols: a contact radius, k_c contact stiffness, E* reduced modulus, R tip
radius, l point-charge depth, g vacuum gap, N mobile-ion density, q = z e,
D diffusivity, beta_11 and beta_33 Vegard coefficients [m^3], Q_11 and Q_12
strain electrostriction [m^4/C^2], d_31, d_33, d_15 piezoelectric [m/V],
Y and nu isotropic elastic constants, eps_r relative permittivity,
chi = eps_0 (eps_r minus 1), omega = 2 pi f, V = V_ac e^{i phi},
U_dc = V_dc minus V_cpd, V_j = eta V.

## 1. Scales (`scales.py`)

Contact (Hertz):
1/E* = (1 minus nu_s^2)/Y_s + (1 minus nu_t^2)/Y_t,
a = (3 F_N R / (4 E*))^(1/3), k_c = 2 E* a.

Screening (dilute Debye, one species):
R_D = sqrt(eps_0 eps_r k_B T / (q^2 N)), omega_M = D / R_D^2.

Voltage divider (static series capacitors, per area):
C_layer = eps_0 eps_layer / h; C_s = eps_0 eps_r / R_D with mobile ions,
eps_0 eps_r / a without; eta = C_layer / (C_layer + C_s); eta = 1 without a
layer. V_j = eta V.

Point-charge depth: l = a / sqrt(3).

Sphere capacitance gradient with dielectric contrast:
C_g = (eps_r minus 1)/(eps_r + 1) times 2 pi eps_0 R^2 / (g (g + R)).
A calibrated value on the instrument card replaces it.

Tests: Hertz number for a known load; R_D for a known density; eta limits
(no layer gives 1, thick layer gives 0); C_g against the sphere formula.

## 2. Vegard response (`mechanisms/vegard.py`)

Model: one mobile species, blocking surface, ideal lattice-gas linear
drift-diffusion, homogeneous half-space unbiased at the card's reference bias,
isotropic elasticity, lab-frame Vegard tensor diag(beta_11, beta_11, beta_33),
prescribed top-hat surface potential V_j inside radius a and zero outside,
traction-free surface, readout at the disk centre.

Responsive density: only an ion next to an empty site can move, so the linear
response uses N_r = N (1 minus N / N_site) = N_site theta (1 minus theta). This
is exact at linear order for an ideal lattice gas. It reduces to N without a
site density (dilute limit). N_r replaces N below, in the prefactor and in R_D.

Dimensionless variables: x = k a, alpha = a / R_D, Omega = omega a^2 / D,
S(x) = sqrt(x^2 + alpha^2 + i Omega) with Re S > 0,
B(x) = [beta_33 (2x + S) + beta_11 (2 nu x + (1 + 2 nu) S)] / (x + S).

Eq. 9a gives the centre displacement at 1w, outward positive, as

u_1 = minus (q N a V_j / (k_B T)) times integral from 0 to infinity of
J_1(x) x B(x) / (x^2 + x S(x) + i Omega) dx.

P0 does not evaluate this integral. It uses its two closed-form limits.
Write B(x) = c + (beta_33 minus beta_11) x / (x + S) with
c = beta_33 + (1 + 2 nu) beta_11.

Low frequency (omega to 0): with 1/(x + S) = (S minus x) / alpha^2 and
the integral of J_1(x) / sqrt(x^2 + alpha^2), equal to (1 minus e^{minus alpha}) / alpha,

I_1 = (alpha + e^{minus alpha} minus 1) / alpha^2 (1/2 for small alpha, 1/alpha for large),
I_2 = (alpha^2 minus 6 + 2 (3 + 3 alpha + alpha^2) e^{minus alpha}) / alpha^4 (1/4, 1/alpha^2),
u_1 = minus (q N a V_j / (k_B T)) [c I_1 + (beta_33 minus beta_11) I_2].

Below alpha = 0.1 the code uses the Taylor series of I_1 and I_2, because
the closed forms cancel to alpha^2 and alpha^4.

High frequency (omega to infinity): the denominator tends to i Omega and
u_1 = minus c q N D V_j / (k_B T a i omega). Amplitude falls as 1/omega and
the phase is minus 90 degrees. The anisotropic part decays faster.

Bridge: the two limits meet at omega_c = D / (a^2 I_1), which is 2 D / a^2
for a much smaller than R_D and D / (a R_D) for a much larger than R_D. With
g = 1 + i omega / omega_c,

u_1 = minus (q N a V_j / (k_B T)) [c I_1 + (beta_33 minus beta_11) I_2 / g] / g.

This form is exact in both limits. For 0.03 < omega/omega_c < 30 it is a
single-relaxation bridge, not Eq. 9a, and the report is approximate. An
offline comparison with the full integral (0.01 <= alpha <= 100, several
beta ratios) gave these deviations:
- below 0.03: within 6 % for alpha >= 1 and 15 % for alpha < 1;
- above 30: within 12 % and 19 %; above 100: within 3.5 %;
- inside the band: about 20 % for isotropic beta, and up to about 60 %
  (90 % for alpha < 1) when beta_11 and beta_33 nearly cancel.

Signed q is kept: positive ions, positive beta and positive low-frequency
voltage give depletion and contraction.

Checks reported: a/R_D, |q V_j|/k_B T (small-signal, caution at 0.1),
omega/omega_c (approximate inside the bridge band), omega/omega_M,
R_D N^(1/3) (below 1 means a formal scale), occupancy N/N_site (reported;
flagged when the site density is missing and the dilute N is used),
|q eta (U_dc minus U_ref)|/k_B T (approximate up to 1, unsupported above),
diffusivity kind (self expected), surface layer (approximate).

Tests: I_1 and I_2 against their small- and large-alpha limits and across
the series switch; static isotropic limit 2 (1 + nu) beta I_1;
high-frequency limit c / (i Omega); sign for positive ions; zero beta gives
zero; missing ionic block gives unsupported; large DC gives unsupported.

## 3. Piezoresponse (`mechanisms/piezoresponse.py`)

Model: decoupled approximation, isotropic elasticity, transversely isotropic
d with polar axis along z, point-charge field (depth cancels at the centre),
readout at the disk centre.

d_eff = (nu + 1/4) d_31 + 3/4 d_33 + 1/4 d_15, with d_15 in the engineering
convention (twice the full-tensor shear entry).

u_1 = d_eff V_j.

If d is absent and P_0 is nonzero, d is derived from the linearized
electrostriction as stated in `CONVENTIONS.md` and the report says so. A
supplied d must be the bulk incremental coefficient at the reference state,
not an apparent PFM amplitude.

Tests: d_eff for a known triple at a known nu; linearity in V_j; phase equals
the drive phase; absent d and zero P_0 gives unsupported.

## 4. Electrostriction with internal Maxwell stress (`mechanisms/electrostriction.py`)

The quadratic response of a dielectric to its internal field. It has two
parts with one field and one quadratic law. Electrostriction is
material-specific (Q). The Maxwell stress is universal (eps_r, Y, nu). Q is
required, and the node is never Maxwell-only (`CONVENTIONS.md`).

Model: linear induced polarization P = chi E, isotropic strain-Q for the
induced polarization, eps_ij = Q_12 P^2 delta_ij + (Q_11 minus Q_12) P_i P_j,
point-charge field of depth l, isotropic elasticity, zero total incremental
surface traction (internal convention), readout at the disk centre.

Coefficients:
c_11 = (6 minus pi + (4 minus pi) nu) / 8,
c_12 = (10 minus 3 pi + (12 minus 3 pi) nu) / 8,
A_ES = (chi^2 / l) (c_11 Q_11 + c_12 Q_12),
A_M = (eps_0 eps_r / (Y l)) [c_12 / 2 minus (nu + 1/2) c_11].
Both in m/V^2 and reported separately.

Response: the sample sees eta U(t), DC and AC alike, so
u_1 = 2 (A_ES + A_M) eta U_dc V_j = 2 (A_ES + A_M) eta^2 U_dc V and
u_2 = (A_ES + A_M) V_j^2 / 2.

The Maxwell term is compressive for usual nu. Q here is strain-based and
excludes Maxwell loading; a measured effective Q that already contains it
would double count. The P_0 cross term is never evaluated here. With d on
the card it is d's business. Without d and with nonzero P_0, piezoresponse
derives d from Q and P_0.

Checks reported:
- field scale eta (|U_dc| + |V|) / l (caution above 1e8 V/m, linear dielectric);
- induced strain scale (caution above 1e-2);
- surface layer (approximate);
- unscreened field with mobile ions (approximate, the DC field is screened);
- nonzero P_0 (note: its linear term is in piezoresponse);
- eta and U_dc.

Tests: c_11 and c_12 at nu = 0 and 0.3; the factor two of the 1w cross term
and the one half of 2w from a time-domain Fourier projection; sign of A_M;
zero U_dc gives zero 1w and nonzero 2w.

## 5. External electrostatic force (`mechanisms/electrostatic.py`)

Model: local apex capacitance gradient C_g (calibrated, or sphere plus an
explicit vacuum gap), force F(t) = minus C_g U(t)^2 / 2 with
U(t) = U_dc + Re[V e^{i omega t}], outward positive.

The gap is never taken from the surface layer. The apex formula grows as 1/g,
so a nanometre layer, or its series-equivalent h / eps_layer, gives forces
orders of magnitude too large in contact. Without a calibrated C_g or an
explicit gap the result is unsupported.

F_1 = minus C_g U_dc V, F_2 = minus C_g V^2 / 4, both times the force
efficiency factor from the instrument card.

The result is a force in newtons. No cone or lever term, no conversion to
displacement; that is the detector's job.

Checks reported: source of C_g, force efficiency uncalibrated, U_dc. When
the sphere estimate is used: gap over tip radius, and a note when the
material has mobile ions (below the ionic relaxation the sample screens like
a conductor and the contrast factor tends to 1).

Tests: sign (attraction for positive U_dc and in-phase V); F_2 against a
quarter C_g V_ac^2; zero at U_dc = 0 for 1w.

## 6. Detector (`detector.py`)

Inputs: the mechanism reports, plus the material, experiment and instrument
cards. k_c is recomputed from Hertz through `scales.py`; the junction is not
needed.

Model: lumped tip between the lever spring k_l and the contact spring k_c,
one damped mode at the contact resonance shared by displacement and force,
photodiode reads the lever.

Static division: z = (k_c u + F) / (k_l + k_c), per mechanism and harmonic.

Transfer at harmonic h: H(h omega) = 1 / (1 minus (h omega / omega_cr)^2 +
i h omega / (Q omega_cr)); H = 1 when no contact resonance is given.

Signal: y_h = H(h omega) z_h times detector_factor, per mechanism; the total
is the complex sum. Amplitude and phase follow.

Output: `DetectorReport` with the per-mechanism breakdown, the total, H per
harmonic, the omitted (unsupported) mechanisms, and the scales a, k_c, k_l,
omega_cr.

Checks reported:
- Unsupported, with no numbers: nothing supported to sum, a mechanism
  reported twice, a readout that differs from the convention, or shared
  scales (a, eta, U_dc, R_D) that differ between reports or from the
  detector's own a.
- Approximate: one flag per approximate mechanism, so the detector is never
  better than what it sums.
- Notes: k_c / k_l, h omega / omega_cr and whether a harmonic sits inside the
  resonance width, no contact resonance (H = 1), uncalibrated detector_factor.

Tests: gain Q and phase minus 90 degrees at resonance; static division with
k_c much larger than k_l; the 2w channel evaluated at 2 omega; refusal of a
mismatched readout string.

## 7. What the demonstration shows (`notebooks/01_forward_p00.ipynb`)

Cards from `esm_p00/examples.py`: LiCoO2 from the oracle fixture with sources,
an illustrative PZT-like polar dielectric, and an AFM probe with a contact
resonance and an order-of-magnitude C_g.

- The LangGraph graph, drawn: a conditional state step, four parallel
  mechanisms, the detector.
- Frequency sweep on LiCoO2: 2 mV drive in the small-signal window, 10 mV DC
  offset. The Vegard plateau and its 1/f roll-off above f_c, the
  electrostatic 1w term, and the contact resonance.
- Pulse, then off-field relaxation on LiCoO2: occupancy theta and the 1w
  signal against the delay, with tau = a^2 / 4D.
- AC amplitude sweep on the polar dielectric: 1w linear (piezo), 2w quadratic
  (electrostriction and electrostatic), with `valid` and `approximate`
  points marked.

Not yet shown: a V_dc sweep through the CPD, and uncertainty bands from the
card distributions.

## 8. Occupancy state after a DC pulse (`material_state/occupancy.py`)

A state step that runs before the mechanisms. It takes the four cards and a
pulse card (pulse_v, pulse_s, delay_s, readout_s). It returns a new material
card and a `StateReport`. The mechanisms then run unchanged on the new card.

Model: one mobile species on N_site sites, ideal lattice gas, a contact that
exchanges ions with the region under the tip.

theta_0 = N / N_site,
U_p = pulse_v minus V_cpd,
theta_p = 1 / (1 + (1 minus theta_0) / theta_0 exp(q eta U_p / (k_B T))),
the equilibrium occupancy under the tip at the pulse,
tau = a^2 / (4 D),
theta = theta_0 + (theta_p minus theta_0) (1 + delay / tau)^(minus 3/2),
the occupancy at readout.

Positive ions and a positive pulse empty the sites under the tip. The
recovery law is the exact centre value of a Gaussian region of radius a
spreading by diffusion in a half-space whose blocking surface acts as a
mirror. For an ideal lattice gas the diffusion of theta is linear, so the law
holds for large changes. Half of the change is gone at about 0.6 tau and 90 %
at about 3.6 tau.

New card: number_density = theta N_site, and reference_bias_v = the readout
U_dc. The readout is much faster than the recovery, so the state is frozen
while it is read and the frozen card is the reference at the readout DC.
Vegard then responds with N_r = N_site theta (1 minus theta). On-field readout
is delay 0 with the readout DC equal to the pulse.

Checks reported:
- no ionic block or no site density: unsupported;
- theta or 1 minus theta below 1e-12: unsupported (nothing left to respond);
- pulse_s D / a^2 below 1: approximate (equilibrium not reached; build-up not
  modelled);
- readout window over tau above 0.1: approximate (the state changes while it
  is read); without a window this is flagged as assumed;
- readout window times f below 10: approximate (too few AC cycles for a 1w
  phasor). Together the two ask for f much greater than 1/tau;
- surface layer: approximate (eta from the untouched card);
- diffusivity kind (chemical expected);
- tau, delay over tau, eta.

Not covered at P0: build-up during short pulses; the shape of the changed
region (the centre value is used as uniform over the probed volume);
reaction kinetics at the contact; ion-ion interactions and phase separation;
pulse trains with memory.

Tests: theta_p limits (zero pulse gives theta_0; large positive and negative
pulses go to 0 and 1); recovery equal to 1 at delay 0 and to 2^(-3/2) at
delay = tau; on-field Vegard ratio theta (1 minus theta) / (theta_0 (1 minus
theta_0)) at high frequency; unsupported without a site density; the detector
inherits an unsupported state.
