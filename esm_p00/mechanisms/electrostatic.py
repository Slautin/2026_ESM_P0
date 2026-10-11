"""External electrostatic force on the tip apex. A force in newtons, not a displacement.

Model
    F(t) = -1/2 C_g U(t)^2,   U(t) = U_dc + Re[V e^{i omega t}],   U_dc = V_dc - V_cpd
    F_1w = -C_g U_dc V        F_2w = -1/4 C_g V^2        both times force_efficiency_factor
Sign: outward positive, so attraction is negative. For positive U_dc and an
in-phase drive the 1w force is negative; the 2w force is negative at phase 2*phi.
C_g is the magnitude of the capacitance gradient dC/dz at the operating point:
a calibrated value from the instrument card, else the sphere-apex estimate with
the dielectric contrast factor and an explicit vacuum gap (docs/PHYSICS.md section 5).
A surface layer is never used as the gap: the apex formula diverges as 1/g.

Reference: Hudlet et al., Eur. Phys. J. B 2, 5 (1998) for the apex capacitance;
oracle P00 Patch 4 for the harmonic factors and the sign convention.
"""
from __future__ import annotations

from esm_p00.cards import ExperimentCard, InstrumentCard, JunctionCard, MaterialCard
from esm_p00.report import Check, MechanismReport
from esm_p00.scales import capacitance_gradient, dc_offset, drive_phasor

NAME = "electrostatic"
OBSERVABLE = "force"
REFERENCE = "Hudlet 1998 apex capacitance; oracle P00 Patch 4 harmonics"

REQUIRES = {
    "material": ["eps_r"],
    "experiment": ["vac_peak_v", "vdc_v", "drive_phase_rad"],
    "instrument": ["tip_radius_m", "force_efficiency_factor",
                   "capacitance_gradient_f_m (optional, preferred)"],
    "junction": ["v_cpd_v", "gap_m (needed without a calibrated C_g)"],
}

ASSUMPTIONS = [
    "local apex capacitor: force from the capacitance gradient at the operating point",
    "sample as a dielectric half-space, contrast factor (eps_r - 1)/(eps_r + 1); a conductor gives 1; "
    "not used when C_g is calibrated",
    "voltage across the whole junction: applied V_ac and U_dc = V_dc - V_cpd, no layer divider",
    "quadratic law expanded by harmonic balance; the DC load is not reported",
    "no cone or lever term; a distributed lever force enters only through force_efficiency_factor",
    "CPD from the junction card; its default 0 is an assumption, not a measurement",
]


def compute(material: MaterialCard, experiment: ExperimentCard,
            instrument: InstrumentCard, junction: JunctionCard) -> MechanismReport:
    # ---------------------------------------------------------- 1. inputs and checks
    eps_r = material.eps_r.value
    tip_radius = instrument.tip_radius_m.value
    calibrated = None if instrument.capacitance_gradient_f_m is None else instrument.capacitance_gradient_f_m.value
    gap = None if junction.gap_m is None else junction.gap_m.value

    found = capacitance_gradient(calibrated=calibrated, tip_radius=tip_radius, gap=gap, eps_r=eps_r)
    if found is None:
        return MechanismReport.unsupported(
            mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
            reasons=["no capacitance gradient: give a calibrated C_g or an explicit vacuum gap; "
                     "a surface layer is not a gap"],
        )
    c_g, c_g_source = found
    u_dc = dc_offset(vdc=experiment.vdc_v, v_cpd=junction.v_cpd_v.value)
    v = drive_phasor(vac_peak=experiment.vac_peak_v, drive_phase=experiment.drive_phase_rad)

    checks = [
        Check.flag("C_g_calibrated", ok=c_g_source == "calibrated",
                   note="C_g is a sphere-plus-gap estimate: apex only, lower bound without cone and lever"),
        Check.flag("force_efficiency_calibrated", ok=instrument.force_efficiency_factor != 1.0,
                   note="force_efficiency_factor is 1.0 (uncalibrated): lever force treated as a point force at the tip"),
        Check.info("U_dc_v", u_dc),
    ]
    scales_used = {"C_g_f_m": c_g, "U_dc_v": u_dc}
    if c_g_source == "sphere_gap":
        checks.append(Check.info("gap_over_tip_radius", gap / tip_radius))
        checks.append(Check.flag("dielectric_contrast", ok=not material.has_mobile_ions,
                                 note="mobile ions present: below the ionic relaxation the sample screens like "
                                      "a conductor and the contrast factor tends to 1"))
        scales_used["gap_m"] = gap

    # ---------------------------------------------------------- 2. calculation
    efficiency = instrument.force_efficiency_factor
    f_1w = -c_g * u_dc * v * efficiency
    f_2w = -0.25 * c_g * v * v * efficiency

    # ---------------------------------------------------------- 3. report
    return MechanismReport.build(
        mechanism=NAME, observable=OBSERVABLE, reference=REFERENCE,
        harmonics={"1w": f_1w, "2w": f_2w},
        checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
    )


def node(state: dict) -> dict:
    """Graph wrapper: read the four cards from the state, append one report."""
    report = compute(state["material"], state["experiment"], state["instrument"], state["junction"])
    return {"reports": [report]}
