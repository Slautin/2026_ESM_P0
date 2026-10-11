"""Occupancy under the tip after a DC pulse: lattice-gas equilibrium, then diffusive recovery.

Model (one mobile species on N_site sites, ideal lattice gas, a contact that exchanges ions)
    theta_0 = N / N_site                                                 untouched card
    U_p     = pulse_v - V_cpd                                            pulse DC offset, junction level
    theta_p = 1 / (1 + (1 - theta_0) / theta_0 exp(q eta U_p / k_B T))   equilibrium under the tip
    tau     = a^2 / (4 D)                                                recovery time of a region of size a
    theta   = theta_0 + (theta_p - theta_0) (1 + delay / tau)^(-3/2)     occupancy at readout
Positive ions and a positive pulse empty the sites under the tip (theta_p < theta_0).
The recovery law is the exact centre value of a Gaussian region of radius a spreading by
diffusion in a half-space whose blocking surface acts as a mirror. Diffusion of theta is
linear for an ideal lattice gas, so the law holds for large changes too.

Output: a copy of the material card with number_density = theta N_site and
reference_bias_v = the readout U_dc, and a StateReport. The readout is assumed much
faster than the recovery, so the state is frozen while it is read, and the frozen card
is the reference at the readout DC. The mechanisms then run on the new card unchanged.
On-field readout is delay_s = 0 with the readout DC equal to the pulse.
"""
from __future__ import annotations

import math

from esm_p00.cards import ExperimentCard, InstrumentCard, IonicBlock, JunctionCard, MaterialCard, PulseCard
from esm_p00.constants import BOLTZMANN, ELEMENTARY_CHARGE
from esm_p00.report import Check, StateReport, status_from_checks
from esm_p00.scales import (dc_offset, debye_scales, divider_length, hertz_contact, reduced_modulus,
                            responsive_density, voltage_fraction)

NAME = "occupancy"

EQUILIBRATION_LIMIT = 1.0   # pulse_s D / a^2: below this the pulse did not reach equilibrium
FROZEN_LIMIT = 0.1          # readout_s / tau: above this the state changes while it is read
READOUT_CYCLES = 10.0       # readout_s f: below this the window holds too few AC cycles for a 1w phasor
THETA_FLOOR = 1e-12         # min(theta, 1 - theta): below this nothing is left to respond
EXP_CAP = 700.0             # exponent cap; beyond it theta is at the floor anyway

REQUIRES = {
    "material": ["ionic.number_density, site_density, diffusivity, charge_number, diffusivity_kind",
                 "eps_r", "youngs_modulus", "poisson_ratio"],
    "experiment": ["vdc_v (readout DC)", "frequency_hz (readout cycles)", "temperature_k", "normal_load_n"],
    "instrument": ["tip_radius_m", "tip_young_modulus_pa", "tip_poisson_ratio"],
    "junction": ["v_cpd_v", "surface layer (optional)"],
    "pulse": ["pulse_v", "pulse_s", "delay_s", "readout_s (optional)"],
}

ASSUMPTIONS = [
    "one mobile species on a fixed site lattice; ideal lattice gas: no interactions, no phase separation",
    "the contact exchanges ions, so the occupancy under the tip reaches equilibrium with the pulse",
    "the pulse reached that equilibrium; the build-up during the pulse is not modelled",
    "after the pulse a Gaussian region of radius a recovers by diffusion with the card D; blocking surface as mirror",
    "the centre occupancy is used as uniform over the probed volume",
    "readout much faster than the recovery: the state is frozen and its pull back toward theta_0 is ignored",
    "eta for the pulse from the static divider of the untouched card",
    "only the ion density changes; D, beta, eps_r, Y and nu keep their card values",
    "the new density is a nominal value; its distribution is not propagated",
]


def compute(material: MaterialCard, experiment: ExperimentCard, instrument: InstrumentCard,
            junction: JunctionCard, pulse: PulseCard) -> tuple[MaterialCard, StateReport]:
    # ---------------------------------------------------------- 1. inputs and checks
    ion = material.ionic
    if ion is None or ion.site_density is None:
        return material, StateReport.unsupported(
            step=NAME, reasons=["occupancy needs an ionic block with a site density; missing data are not zero"],
        )
    density, site, diffusivity = ion.number_density.value, ion.site_density.value, ion.diffusivity.value
    eps_r = material.eps_r.value
    q = ion.charge_number * ELEMENTARY_CHARGE
    kt = BOLTZMANN * experiment.temperature_k

    e_star = reduced_modulus(y_sample=material.youngs_modulus.value, nu_sample=material.poisson_ratio.value,
                             y_tip=instrument.tip_young_modulus_pa.value, nu_tip=instrument.tip_poisson_ratio.value)
    a, _ = hertz_contact(normal_load=experiment.normal_load_n, tip_radius=instrument.tip_radius_m.value, e_star=e_star)
    r_d, _ = debye_scales(charge_number=ion.charge_number,
                          number_density=responsive_density(number_density=density, site_density=site),
                          diffusivity=diffusivity, eps_r=eps_r, temperature=experiment.temperature_k)
    layer = junction.surface_layer_thickness_m
    layer_eps = junction.surface_layer_relative_permittivity
    eta = voltage_fraction(
        layer_thickness=None if layer is None else layer.value,
        layer_eps_r=None if layer_eps is None else layer_eps.value,
        eps_r=eps_r,
        sample_length=divider_length(has_mobile_ions=True, contact_radius=a, debye_length=r_d),
    )
    u_pulse = dc_offset(vdc=pulse.pulse_v, v_cpd=junction.v_cpd_v.value)
    u_read = dc_offset(vdc=experiment.vdc_v, v_cpd=junction.v_cpd_v.value)
    tau = a * a / (4.0 * diffusivity)
    on_field = pulse.delay_s == 0.0 and u_read == u_pulse

    checks = [
        Check.lower("pulse_D_over_a2", pulse.pulse_s * diffusivity / (a * a), EQUILIBRATION_LIMIT),
        Check.flag("no_surface_layer", ok=not junction.has_surface_layer, on_fail="approximate",
                   note="eta for the pulse from the static divider of the untouched card"),
        Check.flag("diffusivity_chemical", ok=ion.diffusivity_kind == "chemical",
                   note=f"diffusivity kind is '{ion.diffusivity_kind}': the recovery is chemical diffusion"),
        Check.info("tau_s", tau),
        Check.info("delay_over_tau", pulse.delay_s / tau),
        Check.info("eta", eta),
    ]
    if not on_field and pulse.readout_s is not None:
        checks.append(Check.upper("readout_over_tau", pulse.readout_s / tau, FROZEN_LIMIT))
        checks.append(Check.lower("readout_cycles", pulse.readout_s * experiment.frequency_hz, READOUT_CYCLES))
    elif not on_field:
        checks.append(Check.flag("readout_window_known", ok=False,
                                 note="no readout window given: a frozen state during readout is assumed, not checked"))

    # ---------------------------------------------------------- 2. calculation
    theta_0 = density / site
    exponent = q * eta * u_pulse / kt + math.log((1.0 - theta_0) / theta_0)
    theta_p = 1.0 / (1.0 + math.exp(min(exponent, EXP_CAP)))
    recovery = (1.0 + pulse.delay_s / tau) ** -1.5
    theta = theta_0 + (theta_p - theta_0) * recovery

    checks.append(Check.lower("theta_margin", min(theta, 1.0 - theta), THETA_FLOOR, on_fail="unsupported"))
    scales_used = {"a_m": a, "R_D_m": r_d, "eta": eta, "tau_s": tau}
    if status_from_checks(checks) == "unsupported":
        return material, StateReport.unsupported(
            step=NAME, reasons=["the region under the tip is emptied or filled: no ions or no free sites left to respond"],
            checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
        )

    # ---------------------------------------------------------- 3. report
    new_ion = IonicBlock(**{**ion.model_dump(), "number_density": theta * site, "reference_bias_v": u_read})
    new_material = MaterialCard(**{**material.model_dump(), "ionic": new_ion})
    return new_material, StateReport.build(
        step=NAME,
        values={"theta_0": theta_0, "theta_pulse": theta_p, "theta": theta,
                "U_pulse_v": u_pulse, "U_readout_v": u_read},
        checks=checks, assumptions=ASSUMPTIONS, scales_used=scales_used,
    )


def node(state: dict) -> dict:
    """Graph wrapper: replace the material card with the prepared one, write the state report."""
    material, report = compute(state["material"], state["experiment"], state["instrument"],
                               state["junction"], state["pulse"])
    return {"material": material, "state_report": report}
