"""Piezoresponse at the disk centre: linear strain from the junction voltage.

Model (decoupled approximation, isotropic elasticity, transversely isotropic d
with polar axis along z, point-charge field of depth l; l cancels at the centre)
    d_eff = (nu + 1/4) d_31 + 3/4 d_33 + 1/4 d_15        d_15 in the engineering convention
    u_1w  = d_eff V_j,     V_j = eta V,     V = V_ac e^{i phi}
Outward positive. Linear in the drive, so there is no 2w term.

Electromechanical rule (docs/CONVENTIONS.md): the piezo block supplies d when it
exists. Without it, a nonzero spontaneous polarization P_0 with isotropic Q gives
the linearized electrostrictive coefficients
    d_33 = 2 chi Q_11 P_0,   d_31 = 2 chi Q_12 P_0,   d_15 = 2 chi (Q_11 - Q_12) P_0,
    chi = eps_0 (eps_r - 1),
and the report says so. The electrostriction node never counts that term again.

Reference: Kalinin, Karapetian, Kachanov, doi:10.1063/1.2206992 (centre coefficients);
oracle P00 Patch 4 for the lab-frame reduction.
"""
from __future__ import annotations

from esm_p00.cards import ExperimentCard, InstrumentCard, JunctionCard, MaterialCard
from esm_p00.constants import VACUUM_PERMITTIVITY
from esm_p00.report import Check, MechanismReport
from esm_p00.scales import (dc_offset, debye_scales, divider_length, drive_phasor, hertz_contact,
                            point_charge_depth, reduced_modulus, responsive_density, voltage_fraction)

NAME = "piezoresponse"
OBSERVABLE = "displacement"
REFERENCE = "Kalinin et al. 2006 doi:10.1063/1.2206992; oracle P00 Patch 4"

REQUIRES = {
    "material": ["poisson_ratio", "eps_r", "youngs_modulus",
                 "piezo.d_31, d_33, d_15  or  electrostriction.Q_11, Q_12, P_0 (P_0 nonzero)",
                 "ionic block if present (divider length only)"],
    "experiment": ["vac_peak_v", "drive_phase_rad", "vdc_v", "normal_load_n", "temperature_k"],
    "instrument": ["tip_radius_m", "tip_young_modulus_pa", "tip_poisson_ratio"],
    "junction": ["v_cpd_v", "surface layer (optional)"],
}

ASSUMPTIONS = [
    "decoupled approximation: the electric field is not changed by the strain",
    "isotropic elasticity with the card Poisson ratio; d transversely isotropic, polar axis along z",
    "point-charge field; the centre displacement does not depend on the field depth",
    "junction voltage V_j = eta V through the static series-layer divider; V_j = V without a layer",
    "bulk incremental d at the reference state: no bias dependence, no domain switching",
    "linear response: 1w only; a measured PFM amplitude is not the required bulk d",
    "readout at the disk centre of a free surface, outward positive",
]


def compute(material: MaterialCard, experiment: ExperimentCard,
            instrument: InstrumentCard, junction: JunctionCard) -> MechanismReport:
    # ---------------------------------------------------------- 1. inputs and checks
    nu = material.poisson_ratio.value
    eps_r = material.eps_r.value

    if material.piezo is not None:
        d_31, d_33, d_15 = (material.piezo.d_31.value, material.piezo.d_33.value, material.piezo.d_15.value)
        d_source = "measured"
    elif material.electrostriction is not None and material.electrostriction.P_0.value != 0.0:
        q_11, q_12, p_0 = (material.electrostriction.Q_11.value, material.electrostriction.Q_12.value,
                           material.electrostriction.P_0.value)
        chi = VACUUM_PERMITTIVITY * (eps_r - 1.0)
        d_33, d_31, d_15 = 2 * chi * q_11 * p_0, 2 * chi * q_12 * p_0, 2 * chi * (q_11 - q_12) * p_0
        d_source = "linearized_Q"
    else:
        return MechanismReport.unsupported(
            mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
            reasons=["no piezoelectric coefficients and no spontaneous polarization: nothing linear to evaluate"],
        )

    e_star = reduced_modulus(y_sample=material.youngs_modulus.value, nu_sample=nu,
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
    v_j = eta * drive_phasor(vac_peak=experiment.vac_peak_v, drive_phase=experiment.drive_phase_rad)
    u_dc = dc_offset(vdc=experiment.vdc_v, v_cpd=junction.v_cpd_v.value)
    depth = point_charge_depth(contact_radius=a)

    checks = [
        Check.flag("d_measured", ok=d_source == "measured",
                   note="d derived from Q and P_0 by linearized electrostriction; not a measured coefficient"),
        Check.flag("no_surface_layer", ok=not junction.has_surface_layer, on_fail="approximate",
                   note="voltage reaching the sample from a static planar series-capacitor estimate"),
        Check.flag("field_unscreened", ok=not material.has_mobile_ions,
                   note="mobile ions present: the unscreened point-charge field is used for the piezo term"),
        Check.flag("zero_dc_bias", ok=u_dc == 0.0,
                   note="U_dc is nonzero: d is taken at the reference state, bias dependence not modeled"),
        Check.info("eta", eta),
        Check.info("field_scale_V_m", abs(v_j) / depth),
    ]
    scales_used = {"a_m": a, "eta": eta, "l_m": depth, "U_dc_v": u_dc}
    if r_d is not None:
        scales_used["R_D_m"] = r_d

    # ---------------------------------------------------------- 2. calculation
    d_eff = (nu + 0.25) * d_31 + 0.75 * d_33 + 0.25 * d_15
    u_1w = d_eff * v_j
    scales_used["d_eff_m_V"] = d_eff

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
