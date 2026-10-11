"""Demonstration cards with their sources.

This module is the stage that builds cards, so provenance lives here as comments and in
each card's note. Distributions encode how far a number is trusted: a lognormal
sigma_ln of 0.7 is about a factor of two, 1.0 is about a factor of three. Formulas read
only the nominal values.

LiCoO2 follows the oracle fixture (2026_ESM_oracle/esm_oracle/sample/examples.py).
The polar dielectric and the probe are illustrative, not a specific material or tip.
"""
from __future__ import annotations

from esm_p00.cards import (ElectrostrictionBlock, InstrumentCard, IonicBlock, MaterialCard, Param,
                           PiezoBlock)

# LiCoO2 sources
_REIMERS = "doi:10.1149/1.2221184"     # lattice parameters vs x: site density, Vegard slopes
_HASEGAWA = "doi:10.1039/D0CP04598E"   # tracer diffusivity, SIMS, 293 K
_QU = "doi:10.1002/aenm.201200107"     # nanoindentation Young's modulus
_VARENYK = "osti:1337498"              # ESM theory; permittivity assumed

LICOO2_SITE_DENSITY = 3.1e28           # Li sites per m^3, a = 2.816 A, c = 14.05 A, Z = 3 (Reimers & Dahn)


def licoo2(state_of_charge: float = 0.5) -> MaterialCard:
    """Layered Li_xCoO2 with x = 1 - state_of_charge; mobile species Li+.

    Vegard: positive delta_N means Li added; c contracts (d ln c / dx = -0.045) and
    a expands (d ln a / dx = +0.004), slopes from the endpoint lattice parameters
    (x = 1 to 0.5), so beta_ij = (d ln a_i / dx) / N_site.
    """
    x = 1.0 - state_of_charge
    return MaterialCard(
        name="LiCoO2",
        note=(f"x = {x:.2f}. Site density and Vegard slopes from {_REIMERS}; D is a c-axis tracer value "
              f"from {_HASEGAWA}; Y and nu from {_QU} (nu assumed there); eps_r is a placeholder "
              f"({_VARENYK}), plausible range 5-20."),
        eps_r=Param(value=10.0, dist="lognormal", dist_params={"sigma_ln": 0.7}),           # placeholder
        youngs_modulus=Param(value=174e9, dist="normal", dist_params={"sigma": 30e9}),     # isotropic, polycrystal
        poisson_ratio=0.30,                                                                # assumed in the source
        ionic=IonicBlock(
            charge_number=1,
            number_density=x * LICOO2_SITE_DENSITY,
            site_density=LICOO2_SITE_DENSITY,
            diffusivity=Param(value=1e-16, dist="lognormal", dist_params={"sigma_ln": 1.0}),
            diffusivity_kind="self",                                                       # tracer, 293 K
            vegard_11=0.004 / LICOO2_SITE_DENSITY,                                         # m^3 per ion
            vegard_33=-0.045 / LICOO2_SITE_DENSITY,
        ),
    )


def polar_dielectric() -> MaterialCard:
    """Illustrative PZT-like ferroelectric: d, Q and P_0, no mobile ions."""
    return MaterialCard(
        name="polar dielectric",
        note="Illustrative values of the right order for a soft PZT-like film; not a specific material.",
        eps_r=200.0,
        youngs_modulus=80e9,
        poisson_ratio=0.30,
        piezo=PiezoBlock(d_31=-40e-12, d_33=100e-12, d_15=120e-12),                       # m/V, polar axis along z
        electrostriction=ElectrostrictionBlock(Q_11=0.09, Q_12=-0.03, P_0=0.30),            # m^4/C^2, C/m^2
    )


def afm_probe() -> InstrumentCard:
    """Illustrative conductive Si probe in contact mode with a measured contact resonance."""
    return InstrumentCard(
        tip_radius_m=25e-9,
        tip_young_modulus_pa=169e9,                   # Si
        tip_poisson_ratio=0.27,
        cantilever_stiffness_n_m=3.0,
        contact_resonance_hz=320e3,                   # would be measured on the sample
        contact_quality_factor=80.0,
        capacitance_gradient_f_m=1e-9,                # order of magnitude for tip plus lever; not a calibration
    )
