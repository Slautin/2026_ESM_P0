"""Detector: mechanism displacements and forces -> measured signal per harmonic.

Model (lumped tip between the lever spring k_l and the contact spring k_c, one damped
contact-resonance mode shared by both inputs, photodiode reads the lever)
    z_h   = (k_c u_h + F_h) / (k_l + k_c)                              two springs, outward positive
    H(w)  = 1 / (1 - (w / w_cr)^2 + i w / (Q w_cr)),   H = 1 without a contact resonance
    y_h^m = detector_factor H(h omega) z_h^m                           per mechanism m
    y_h   = sum over m of y_h^m                                        complex sum, never amplitudes
The 2w channel is evaluated at 2 omega. Attraction (F < 0) pulls the tip toward the
sample. With detector_factor = 1.0 the signal is the tip deflection in metres.

This is the only place where mechanisms meet. Unsupported mechanisms are listed as
omitted, never counted as zero. The detector refuses to sum reports whose readouts or
shared scales differ, and it is never better than the mechanisms it sums, nor than the
state step (material_state) that prepared their card, when there is one.

Reference: two-springs and contact-resonance model of the oracle detection stage
(esm_oracle/physics/detection.py); docs/PHYSICS.md section 6.
"""
from __future__ import annotations

import math

from esm_p00.cards import ExperimentCard, InstrumentCard, MaterialCard
from esm_p00.report import READOUT, Check, DetectorReport, MechanismReport, StateReport, status_from_checks
from esm_p00.scales import hertz_contact, reduced_modulus

HARMONIC_ORDER = {"1w": 1, "2w": 2}
SHARED_SCALES = ("a_m", "eta", "U_dc_v", "R_D_m")      # must agree across the summed reports
SCALE_TOLERANCE = 1e-9                                 # relative

REQUIRES = {
    "reports": ["MechanismReport list from the mechanism nodes", "StateReport (optional, from material_state)"],
    "material": ["youngs_modulus", "poisson_ratio"],
    "experiment": ["frequency_hz", "normal_load_n"],
    "instrument": ["tip_radius_m", "tip_young_modulus_pa", "tip_poisson_ratio", "cantilever_stiffness_n_m",
                   "contact_resonance_hz and contact_quality_factor (optional)", "detector_factor"],
}

ASSUMPTIONS = [
    "lumped tip between the lever spring k_l and the Hertz contact spring k_c; vertical motion only",
    "one damped contact-resonance mode, the same transfer for surface displacement and tip force",
    "contact resonance f_c and Q are measured inputs; k_c is computed; the two are not cross-checked",
    "quasi-static transfer H = 1 when no contact resonance is given",
    "linear: contributions add as complex phasors per harmonic with a common time origin",
    "no distributed lever (Euler-Bernoulli), no lateral or buckling modes",
    "detector_factor converts tip deflection to the signal; at 1.0 the signal is tip deflection in metres",
]


def _contact_transfer(omega: float, omega_cr: float | None, quality: float | None) -> complex:
    """H = 1 / (1 - r^2 + i r / Q) with r = omega / omega_cr; 1 without a contact resonance."""
    if omega_cr is None:
        return 1.0 + 0j
    r = omega / omega_cr
    return 1.0 / (1.0 - r * r + 1j * r / quality)


def compute(reports: list[MechanismReport], material: MaterialCard, experiment: ExperimentCard,
            instrument: InstrumentCard, state_report: StateReport | None = None) -> DetectorReport:
    # ---------------------------------------------------------- 1. inputs and checks
    included = [r for r in reports if r.is_supported]
    omitted = [r.mechanism for r in reports if not r.is_supported]
    names = [r.mechanism for r in included]

    e_star = reduced_modulus(y_sample=material.youngs_modulus.value, nu_sample=material.poisson_ratio.value,
                             y_tip=instrument.tip_young_modulus_pa.value, nu_tip=instrument.tip_poisson_ratio.value)
    a, k_c = hertz_contact(normal_load=experiment.normal_load_n, tip_radius=instrument.tip_radius_m.value,
                           e_star=e_star)
    k_l = instrument.cantilever_stiffness_n_m.value
    omega = experiment.omega
    omega_cr = None
    quality = None
    if instrument.has_contact_resonance:
        omega_cr = 2.0 * math.pi * instrument.contact_resonance_hz.value
        quality = instrument.contact_quality_factor.value
    present = [h for h in HARMONIC_ORDER if any(h in r.harmonics for r in included)]

    misread = [r.mechanism for r in included if r.readout != READOUT[r.observable]]
    disagree = []
    for key in SHARED_SCALES:
        values = [r.scales_used[key] for r in included if key in r.scales_used] + ([a] if key == "a_m" else [])
        if any(not math.isclose(v, values[0], rel_tol=SCALE_TOLERANCE) for v in values):
            disagree.append(key)

    checks = [
        Check.flag("any_supported", ok=bool(included), on_fail="unsupported",
                   note="no supported mechanism: nothing to sum"),
        Check.flag("unique_mechanisms", ok=len(set(names)) == len(names), on_fail="unsupported",
                   note="a mechanism was reported more than once"),
        Check.flag("readout_consistent", ok=not misread, on_fail="unsupported",
                   note=f"readout differs from the convention for: {', '.join(misread)}"),
        Check.flag("shared_scales_agree", ok=not disagree, on_fail="unsupported",
                   note=f"mechanisms or detector disagree on {', '.join(disagree)}: the nodes were run on different cards"),
    ]
    checks += [Check.flag(f"{r.mechanism}_valid", ok=r.status == "valid", on_fail="approximate",
                          note=f"{r.mechanism} is approximate: {'; '.join(r.reasons)}") for r in included]
    if state_report is not None:
        checks.append(Check.flag(f"{state_report.step}_valid", ok=state_report.status == "valid",
                                 on_fail="unsupported" if state_report.status == "unsupported" else "approximate",
                                 note=f"state {state_report.step} is {state_report.status}: "
                                      f"{'; '.join(state_report.reasons)}"))
    checks += [
        Check.info("k_c_over_k_l", k_c / k_l),
        Check.flag("detector_calibrated", ok=instrument.detector_factor != 1.0,
                   note="detector_factor is 1.0 (uncalibrated): the signal is tip deflection in metres"),
        Check.flag("contact_resonance_given", ok=omega_cr is not None,
                   note="no contact resonance on the card: quasi-static transfer H = 1"),
    ]
    if omega_cr is not None:
        for h in present:
            omega_h = HARMONIC_ORDER[h] * omega
            checks.append(Check.info(f"{h}_over_resonance", omega_h / omega_cr))
            checks.append(Check.flag(f"{h}_outside_resonance_width", ok=abs(omega_h - omega_cr) >= omega_cr / quality,
                                     note=f"{h} lies inside the resonance width: the signal is very sensitive to f_c and Q"))

    scales_used = {"a_m": a, "k_c_n_m": k_c, "k_l_n_m": k_l}
    if omega_cr is not None:
        scales_used["omega_cr_rad_s"] = omega_cr
    if status_from_checks(checks) == "unsupported":
        return DetectorReport.unsupported(reasons=["contributions may not be summed"], checks=checks,
                                          assumptions=ASSUMPTIONS, scales_used=scales_used, omitted=omitted)

    # ---------------------------------------------------------- 2. calculation
    transfer = {h: _contact_transfer(HARMONIC_ORDER[h] * omega, omega_cr, quality) for h in present}
    gain = instrument.detector_factor / (k_l + k_c)
    per_mechanism = {}
    for r in included:
        stiffness = k_c if r.observable == "displacement" else 1.0       # k_c u for displacement, F for force
        per_mechanism[r.mechanism] = {h: gain * transfer[h] * stiffness * x for h, x in r.harmonics.items()}

    # ---------------------------------------------------------- 3. report
    return DetectorReport.build(
        per_mechanism=per_mechanism, transfer=transfer, omitted=omitted,
        checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
    )


def node(state: dict) -> dict:
    """Graph wrapper: read the reports and cards from the state, write the detector report."""
    report = compute(state["reports"], state["material"], state["experiment"], state["instrument"],
                     state.get("state_report"))
    return {"detector": report}
