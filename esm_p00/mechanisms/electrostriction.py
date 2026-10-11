"""Quadratic response of a dielectric to its internal field, at the disk centre.

Two parts with the same field, the same geometry and the same quadratic law:
electrostriction (material-specific, strain = Q P^2 with P = chi E) and the
dielectric Maxwell stress (universal, about eps_0 eps_r E^2 / Y). Q is required;
without it the node is unsupported, never Maxwell-only. With mobile ions the
field is screened and neither part is a separate bulk effect; the reaction of
the tip-sample attraction is then the external electrostatic force.

Model (linear induced polarization P = chi E, isotropic strain-Q, point-charge
field of depth l, isotropic elasticity, zero total surface traction)
    chi  = eps_0 (eps_r - 1)
    c_11 = (6 - pi + (4 - pi) nu) / 8          c_12 = (10 - 3 pi + (12 - 3 pi) nu) / 8
    A_ES = (chi^2 / l) (c_11 Q_11 + c_12 Q_12)                        electrostriction, m/V^2
    A_M  = (eps_0 eps_r / (Y l)) [c_12 / 2 - (nu + 1/2) c_11]         Maxwell stress, m/V^2
The sample sees the junction voltage U_s(t) = eta [U_dc + Re(V e^{i omega t})], so with
A = A_ES + A_M and harmonic balance of A U_s(t)^2:
    u_1w = 2 A eta^2 U_dc V          u_2w = A eta^2 V^2 / 2
Outward positive. The Maxwell term is compressive for ordinary nu.

Electromechanical rule (docs/CONVENTIONS.md): this node computes only the part
quadratic in the induced polarization. The term linear in a spontaneous P_0 is
the piezoresponse node's business, from a measured d or from the linearized Q.
Q is the bare strain coefficient and excludes Maxwell loading; a measured
effective Q that already contains it would be counted twice.

Reference: Eliseev et al., arXiv:1403.1225 (framework); oracle P00 Patch 4 for
the centre coefficients c_11, c_12 and the Maxwell eigenstrain convention.
"""
from __future__ import annotations

import math

from esm_p00.cards import ExperimentCard, InstrumentCard, JunctionCard, MaterialCard
from esm_p00.constants import VACUUM_PERMITTIVITY
from esm_p00.report import Check, MechanismReport
from esm_p00.scales import (dc_offset, debye_scales, divider_length, drive_phasor, hertz_contact,
                            point_charge_depth, reduced_modulus, responsive_density, voltage_fraction)

NAME = "electrostriction"
OBSERVABLE = "displacement"
REFERENCE = "Eliseev et al. arXiv:1403.1225; oracle P00 Patch 4 coefficients"

FIELD_LIMIT_V_M = 1e8      # above this the linear-dielectric assumption is doubtful
STRAIN_LIMIT = 1e-2        # small-strain assumption

REQUIRES = {
    "material": ["electrostriction.Q_11, Q_12, P_0", "eps_r", "youngs_modulus", "poisson_ratio",
                 "piezo block if present (rule only)", "ionic block if present (divider length only)"],
    "experiment": ["vac_peak_v", "drive_phase_rad", "vdc_v", "normal_load_n", "temperature_k"],
    "instrument": ["tip_radius_m", "tip_young_modulus_pa", "tip_poisson_ratio"],
    "junction": ["v_cpd_v", "surface layer (optional)"],
}

ASSUMPTIONS = [
    "linear induced polarization P = eps_0 (eps_r - 1) E; no field dependence of eps_r",
    "isotropic strain-Q for the induced polarization; Q excludes Maxwell loading",
    "point-charge field of depth l = a / sqrt(3); the field is not changed by the strain",
    "isotropic elasticity with the card Y and nu; zero total incremental traction at the surface",
    "the sample sees eta times the junction voltage, DC and AC alike; eta = 1 without a layer",
    "only the part quadratic in induced polarization; the linear P_0 term lives in piezoresponse",
    "quadratic law expanded by harmonic balance; the DC strain is not reported",
    "readout at the disk centre of a free surface, outward positive",
]


def compute(material: MaterialCard, experiment: ExperimentCard,
            instrument: InstrumentCard, junction: JunctionCard) -> MechanismReport:
    # ---------------------------------------------------------- 1. inputs and checks
    if material.electrostriction is None:
        return MechanismReport.unsupported(
            mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
            reasons=["no electrostriction block: Q_11, Q_12 are required; missing Q is not zero"],
        )
    q_11, q_12, p_0 = (material.electrostriction.Q_11.value, material.electrostriction.Q_12.value,
                       material.electrostriction.P_0.value)
    nu = material.poisson_ratio.value
    y = material.youngs_modulus.value
    eps_r = material.eps_r.value
    chi = VACUUM_PERMITTIVITY * (eps_r - 1.0)

    e_star = reduced_modulus(y_sample=y, nu_sample=nu,
                             y_tip=instrument.tip_young_modulus_pa.value, nu_tip=instrument.tip_poisson_ratio.value)
    a, _ = hertz_contact(normal_load=experiment.normal_load_n, tip_radius=instrument.tip_radius_m.value, e_star=e_star)
    r_d = None
    if material.ionic is not None:
        ion = material.ionic
        n_resp = responsive_density(number_density=ion.number_density.value,
                                    site_density=None if ion.site_density is None else ion.site_density.value)
        r_d, _ = debye_scales(charge_number=ion.charge_number, number_density=n_resp,
                              diffusivity=ion.diffusivity.value, eps_r=eps_r, temperature=experiment.temperature_k)
    layer = junction.surface_layer_thickness_m
    layer_eps = junction.surface_layer_relative_permittivity
    eta = voltage_fraction(
        layer_thickness=None if layer is None else layer.value,
        layer_eps_r=None if layer_eps is None else layer_eps.value,
        eps_r=eps_r,
        sample_length=divider_length(has_mobile_ions=material.has_mobile_ions, contact_radius=a, debye_length=r_d),
    )
    v = drive_phasor(vac_peak=experiment.vac_peak_v, drive_phase=experiment.drive_phase_rad)
    u_dc = dc_offset(vdc=experiment.vdc_v, v_cpd=junction.v_cpd_v.value)
    depth = point_charge_depth(contact_radius=a)

    field_scale = eta * (abs(u_dc) + abs(v)) / depth              # peak field under the apex, V/m
    strain_scale = abs(q_11) * (chi * field_scale) ** 2            # induced electrostrictive strain
    checks = [
        Check.upper("field_scale_V_m", field_scale, FIELD_LIMIT_V_M),
        Check.upper("strain_scale", strain_scale, STRAIN_LIMIT),
        Check.flag("no_surface_layer", ok=not junction.has_surface_layer, on_fail="approximate",
                   note="voltage reaching the sample from a static planar series-capacitor estimate"),
        Check.flag("field_unscreened", ok=not material.has_mobile_ions, on_fail="approximate",
                   note="mobile ions present: the ions screen the DC field, so the unscreened point-charge field "
                        "overestimates the 1w term"),
        Check.flag("nonpolar", ok=p_0 == 0.0,
                   note="P_0 is nonzero: its linear term is evaluated by the piezoresponse node, not here"),
        Check.info("eta", eta),
        Check.info("U_dc_v", u_dc),
    ]

    # ---------------------------------------------------------- 2. calculation
    c_11 = (6.0 - math.pi + (4.0 - math.pi) * nu) / 8.0
    c_12 = (10.0 - 3.0 * math.pi + (12.0 - 3.0 * math.pi) * nu) / 8.0
    a_es = chi**2 / depth * (c_11 * q_11 + c_12 * q_12)
    a_m = VACUUM_PERMITTIVITY * eps_r / (y * depth) * (0.5 * c_12 - (nu + 0.5) * c_11)
    a_total = a_es + a_m
    u_1w = 2.0 * a_total * eta**2 * u_dc * v
    u_2w = 0.5 * a_total * eta**2 * v * v

    # ---------------------------------------------------------- 3. report
    scales_used = {"a_m": a, "eta": eta, "l_m": depth, "U_dc_v": u_dc,
                   "A_ES_m_V2": a_es, "A_M_m_V2": a_m}
    if r_d is not None:
        scales_used["R_D_m"] = r_d
    return MechanismReport.build(
        mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
        harmonics={"1w": u_1w, "2w": u_2w},
        checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
    )


def node(state: dict) -> dict:
    """Graph wrapper: read the four cards from the state, append one report."""
    report = compute(state["material"], state["experiment"], state["instrument"], state["junction"])
    return {"reports": [report]}
