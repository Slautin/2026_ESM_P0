"""Vegard (ionic) response at the disk centre: closed-form limits of Morozovska et al. 2012 Eq. 9a.

Model (one mobile species, blocking surface, ideal lattice-gas linear drift-diffusion
about an unbiased homogeneous half-space, isotropic elasticity, Vegard tensor
diag(beta_11, beta_11, beta_33), prescribed top-hat potential V_j inside radius a,
traction-free surface)
    N_r     = N (1 - N / N_site)                                            responsive density; N without N_site
    alpha   = a / R_D,   R_D from N_r
    I_1     = (alpha + e^{-alpha} - 1) / alpha^2                            1/2 for a << R_D, 1/alpha for a >> R_D
    I_2     = (alpha^2 - 6 + 2 (3 + 3 alpha + alpha^2) e^{-alpha}) / alpha^4  1/4 for a << R_D, 1/alpha^2 for a >> R_D
    c       = beta_33 + (1 + 2 nu) beta_11
    omega_c = D / (a^2 I_1)                                                 2 D / a^2,  D / (a R_D)
    g       = 1 + i omega / omega_c
    u_1w    = -(q N_r a V_j / k_B T) [c I_1 + (beta_33 - beta_11) I_2 / g] / g,   q = z e,  V_j = eta V
Outward positive. Linear in the drive, so there is no 2w term. Only an ion next to an
empty site can move, so the linear response uses N_r = N_site theta (1 - theta); this is
exact for an ideal lattice gas and reduces to N in the dilute limit.

Limits, both exact for Eq. 9a:
    omega << omega_c   equilibrium screening under the tip,
                       u_1w = -(q N_r a V_j / k_B T) [c I_1 + (beta_33 - beta_11) I_2]
    omega >> omega_c   transport-limited, u_1w = -c q N_r D V_j / (k_B T a i omega):
                       amplitude falls as 1/omega, phase -90 degrees
Between them g is a single-relaxation bridge, not Eq. 9a; the report is approximate
there. The full integral needs quadrature and is not part of P0.

Sign: positive ions, positive beta and a positive low-frequency voltage deplete the
region under the tip, and the surface contracts (u_1w < 0, in phase with V).

Bias rule (docs/CONVENTIONS.md): linear response about the card's reference state,
which is unbiased at U_ref = ionic.reference_bias_v (0 for an untouched card; set by
material_state). The sample sees eta (U_dc - U_ref), like every strain mechanism.
|q eta (U_dc - U_ref)| / k_B T up to 1 is approximate; above it the report is
unsupported and the biased state belongs to material_state/.

Reference: Morozovska, Eliseev, Kalinin, J. Appl. Phys. 111, 014107 (2012),
doi:10.1063/1.3675508, Eq. 9a; static anisotropic and high-frequency limits derived here.
"""
from __future__ import annotations

import math

from esm_p00.cards import ExperimentCard, InstrumentCard, JunctionCard, MaterialCard
from esm_p00.constants import BOLTZMANN, ELEMENTARY_CHARGE
from esm_p00.report import Check, MechanismReport, status_from_checks
from esm_p00.scales import (dc_offset, debye_scales, divider_length, drive_phasor, hertz_contact,
                            reduced_modulus, responsive_density, voltage_fraction)

NAME = "vegard"
OBSERVABLE = "displacement"
REFERENCE = "Morozovska, Eliseev, Kalinin 2012 doi:10.1063/1.3675508 Eq. 9a, closed-form limits"

SMALL_SIGNAL_LIMIT = 0.1   # |q V_j| / k_B T: linearized drift-diffusion
BIAS_LIMIT = 1.0           # |q eta (U_dc - U_ref)| / k_B T: beyond this the reference state fails
CONTINUUM_LIMIT = 1.0      # R_D N^(1/3): below this R_D is shorter than the ion spacing
BRIDGE_LOW = 0.03          # omega / omega_c at or below: low-frequency limit holds
BRIDGE_HIGH = 30.0         # omega / omega_c at or above: high-frequency limit holds
SERIES_BELOW = 0.1         # alpha below which I_1, I_2 use their Taylor series

REQUIRES = {
    "material": ["ionic.charge_number, number_density, diffusivity, vegard_11, vegard_33",
                 "ionic.site_density (optional: lattice-gas responsive density)", "ionic.diffusivity_kind",
                 "ionic.reference_bias_v (0 unless prepared by material_state)",
                 "eps_r", "youngs_modulus", "poisson_ratio"],
    "experiment": ["vac_peak_v", "drive_phase_rad", "frequency_hz", "vdc_v", "normal_load_n", "temperature_k"],
    "instrument": ["tip_radius_m", "tip_young_modulus_pa", "tip_poisson_ratio"],
    "junction": ["v_cpd_v", "surface layer (optional)"],
}

ASSUMPTIONS = [
    "one dominant mobile ionic species, blocking surface, ideal lattice-gas linear drift-diffusion",
    "responsive density N (1 - N / N_site); without a site density the dilute limit N is used",
    "linear response about a homogeneous reference state, unbiased at U_ref; no stress feedback on the ions",
    "semi-infinite sample, quasistatic traction-free surface, isotropic elasticity with the card nu",
    "lab-frame Vegard tensor diag(beta_11, beta_11, beta_33), strain_ij = beta_ij delta_N",
    "prescribed top-hat potential V_j inside radius a, zero outside; not the equipotential-disk mixed problem",
    "closed-form low- and high-frequency limits of Eq. 9a, joined by a single relaxation at omega_c",
    "junction voltage V_j = eta V through the static series-layer divider; V_j = V without a layer",
    "with a layer, eta uses the screening capacitance eps_0 eps_r / R_D: a low-frequency estimate above omega_c",
    "CPD from the junction card; its default 0 is an assumption, not a measurement",
    "readout at the disk centre of a free surface, outward positive",
]


def _screening_integrals(alpha: float) -> tuple[float, float]:
    """(I_1, I_2): the static disk integrals of Eq. 9a in closed form. Pure numbers."""
    if alpha < SERIES_BELOW:          # the closed forms cancel to alpha^2 and alpha^4 here
        i_1 = sum((-alpha) ** n / math.factorial(n + 2) for n in range(6))
        i_2 = sum(2.0 * (-alpha) ** n * (n + 1) * (n + 3) / math.factorial(n + 4) for n in range(6))
        return i_1, i_2
    e = math.exp(-alpha)
    return (alpha + e - 1.0) / alpha**2, (alpha**2 - 6.0 + 2.0 * (3.0 + 3.0 * alpha + alpha**2) * e) / alpha**4


def compute(material: MaterialCard, experiment: ExperimentCard,
            instrument: InstrumentCard, junction: JunctionCard) -> MechanismReport:
    # ---------------------------------------------------------- 1. inputs and checks
    if material.ionic is None:
        return MechanismReport.unsupported(
            mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
            reasons=["no ionic block: one mobile species with N, D and Vegard coefficients is required; "
                     "missing data are not zero strain"],
        )
    ion = material.ionic
    beta_11, beta_33 = ion.vegard_11.value, ion.vegard_33.value
    density, diffusivity = ion.number_density.value, ion.diffusivity.value
    nu = material.poisson_ratio.value
    eps_r = material.eps_r.value
    q = ion.charge_number * ELEMENTARY_CHARGE
    kt = BOLTZMANN * experiment.temperature_k

    e_star = reduced_modulus(y_sample=material.youngs_modulus.value, nu_sample=nu,
                             y_tip=instrument.tip_young_modulus_pa.value, nu_tip=instrument.tip_poisson_ratio.value)
    a, _ = hertz_contact(normal_load=experiment.normal_load_n, tip_radius=instrument.tip_radius_m.value, e_star=e_star)
    n_resp = responsive_density(number_density=density,
                                site_density=None if ion.site_density is None else ion.site_density.value)
    r_d, omega_m = debye_scales(charge_number=ion.charge_number, number_density=n_resp, diffusivity=diffusivity,
                                eps_r=eps_r, temperature=experiment.temperature_k)
    layer = junction.surface_layer_thickness_m
    layer_eps = junction.surface_layer_relative_permittivity
    eta = voltage_fraction(
        layer_thickness=None if layer is None else layer.value,
        layer_eps_r=None if layer_eps is None else layer_eps.value,
        eps_r=eps_r,
        sample_length=divider_length(has_mobile_ions=material.has_mobile_ions, contact_radius=a, debye_length=r_d),
    )
    v_j = eta * drive_phasor(vac_peak=experiment.vac_peak_v, drive_phase=experiment.drive_phase_rad)
    u_dc = dc_offset(vdc=experiment.vdc_v, v_cpd=junction.v_cpd_v.value)
    bias = u_dc - ion.reference_bias_v                            # DC not already absorbed by the state
    alpha = a / r_d
    i_1, i_2 = _screening_integrals(alpha)
    omega_c = diffusivity / (a * a * i_1)
    ratio = experiment.omega / omega_c

    occupancy = (Check.info("occupancy", density / ion.site_density.value)
                 if ion.site_density is not None else
                 Check.flag("occupancy_known", ok=False,
                            note="no site density on the card: the dilute limit N is used as the responsive density"))
    checks = [
        Check.upper("qV_j_over_kT", abs(q * v_j) / kt, SMALL_SIGNAL_LIMIT),
        Check.flag("unbiased_reference", ok=bias == 0.0, on_fail="approximate",
                   note="U_dc differs from the reference bias: linear response kept only in the small-bias window"),
        Check.upper("q_eta_bias_over_kT", abs(q * eta * bias) / kt, BIAS_LIMIT, on_fail="unsupported"),
        Check.flag("frequency_limit", ok=ratio <= BRIDGE_LOW or ratio >= BRIDGE_HIGH, on_fail="approximate",
                   note="omega near omega_c: single-relaxation bridge between the exact limits, not Eq. 9a itself"),
        Check.lower("R_D_N13", r_d * density ** (1.0 / 3.0), CONTINUUM_LIMIT),
        occupancy,
        Check.flag("diffusivity_self", ok=ion.diffusivity_kind == "self",
                   note=f"diffusivity kind is '{ion.diffusivity_kind}': the dilute formula expects the self diffusivity"),
        Check.flag("no_surface_layer", ok=not junction.has_surface_layer, on_fail="approximate",
                   note="voltage reaching the sample from a static planar series-capacitor estimate"),
        Check.info("a_over_R_D", alpha),
        Check.info("omega_over_omega_c", ratio),
        Check.info("omega_over_omega_M", experiment.omega / omega_m),
        Check.info("eta", eta),
        Check.info("U_dc_v", u_dc),
        Check.info("U_ref_v", ion.reference_bias_v),
    ]
    scales_used = {"a_m": a, "R_D_m": r_d, "omega_c_rad_s": omega_c, "eta": eta, "U_dc_v": u_dc}
    if status_from_checks(checks) == "unsupported":
        return MechanismReport.unsupported(
            mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
            reasons=["outside the small-bias window of the unbiased response; the biased state belongs to material_state/"],
            checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
        )

    # ---------------------------------------------------------- 2. calculation
    c = beta_33 + (1.0 + 2.0 * nu) * beta_11
    g = 1.0 + 1j * ratio
    u_1w = -(q * n_resp * a / kt) * (c * i_1 + (beta_33 - beta_11) * i_2 / g) / g * v_j

    # ---------------------------------------------------------- 3. report
    return MechanismReport.build(
        mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
        harmonics={"1w": u_1w},
        checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
    )


def node(state: dict) -> dict:
    """Graph wrapper: read the four cards from the state, append one report."""
    report = compute(state["material"], state["experiment"], state["instrument"], state["junction"])
    return {"reports": [report]}
