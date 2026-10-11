"""Shared derived scales: the single source of these formulas.

Numbers in, numbers out, keyword arguments only. No card imports, no status,
no warnings: each mechanism decides what a number means for its own
applicability. Formulas are documented in docs/PHYSICS.md section 1 and the
conventions they encode in docs/CONVENTIONS.md.

Every mechanism and the detector call these functions and never copy them.
"""
from __future__ import annotations

import cmath
import math

from esm_p00.constants import BOLTZMANN, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")


# ---------------------------------------------------------------- contact

def reduced_modulus(*, y_sample: float, nu_sample: float, y_tip: float, nu_tip: float) -> float:
    """1/E* = (1 - nu_s^2)/Y_s + (1 - nu_t^2)/Y_t. Pa."""
    _positive(y_sample=y_sample, y_tip=y_tip)
    return 1.0 / ((1.0 - nu_sample**2) / y_sample + (1.0 - nu_tip**2) / y_tip)


def hertz_contact(*, normal_load: float, tip_radius: float, e_star: float) -> tuple[float, float]:
    """Hertz sphere on flat: a = (3 F R / 4 E*)^(1/3), k_c = 2 E* a. Returns (a [m], k_c [N/m])."""
    _positive(normal_load=normal_load, tip_radius=tip_radius, e_star=e_star)
    a = (3.0 * normal_load * tip_radius / (4.0 * e_star)) ** (1.0 / 3.0)
    return a, 2.0 * e_star * a


# -------------------------------------------------------------- screening

def responsive_density(*, number_density: float, site_density: float | None) -> float:
    """Ions able to respond: N (1 - N / N_site) for an ideal lattice gas, N without a site density. m^-3.

    Only an ion next to an empty site can move, so N_site theta (1 - theta) replaces N in
    the linear response: screening length, Vegard prefactor, conductivity. Dilute limit: N.
    """
    _positive(number_density=number_density)
    if site_density is None:
        return number_density
    _positive(site_density=site_density)
    if number_density >= site_density:
        raise ValueError("number_density must be below site_density")
    return number_density * (1.0 - number_density / site_density)


def debye_scales(*, charge_number: int, number_density: float, diffusivity: float,
                 eps_r: float, temperature: float) -> tuple[float, float]:
    """Single-species scales: R_D = sqrt(eps0 eps_r kT / q^2 N), omega_M = D / R_D^2.

    N is the responsive density from `responsive_density`; callers never pass the raw
    density. Returns (R_D [m], omega_M [rad/s]). Uses q squared; the sign of the ion
    charge is the caller's business (Vegard keeps it in its prefactor).
    """
    _positive(number_density=number_density, diffusivity=diffusivity, eps_r=eps_r, temperature=temperature)
    if charge_number == 0:
        raise ValueError("charge_number must be nonzero")
    q = charge_number * ELEMENTARY_CHARGE
    r_d = math.sqrt(VACUUM_PERMITTIVITY * eps_r * BOLTZMANN * temperature / (q * q * number_density))
    return r_d, diffusivity / (r_d * r_d)


# ------------------------------------------------------------- junction voltage

def divider_length(*, has_mobile_ions: bool, contact_radius: float, debye_length: float | None) -> float:
    """Length that sets the sample capacitance per area in the layer divider.

    R_D when the sample has mobile ions (screening capacitance), the contact
    radius a otherwise (disk-field scale; the exact disk gives a factor 4/pi,
    not applied). One rule, one place.
    """
    if has_mobile_ions:
        if debye_length is None:
            raise ValueError("a material with mobile ions needs its Debye length for the divider")
        return debye_length
    return contact_radius


def voltage_fraction(*, layer_thickness: float | None, layer_eps_r: float | None,
                     eps_r: float, sample_length: float) -> float:
    """Static series-capacitor divider: eta = C_layer / (C_layer + C_sample), 1 without a layer.

    C_layer = eps0 eps_layer / h and C_sample = eps0 eps_r / sample_length,
    both per unit area.
    """
    if (layer_thickness is None) != (layer_eps_r is None):
        raise ValueError("layer thickness and permittivity must be given together")
    if layer_thickness is None:
        return 1.0
    _positive(layer_thickness=layer_thickness, layer_eps_r=layer_eps_r, eps_r=eps_r, sample_length=sample_length)
    c_layer = VACUUM_PERMITTIVITY * layer_eps_r / layer_thickness
    c_sample = VACUUM_PERMITTIVITY * eps_r / sample_length
    return c_layer / (c_layer + c_sample)


def drive_phasor(*, vac_peak: float, drive_phase: float = 0.0) -> complex:
    """Complex peak drive V = V_ac exp(i phi) for V(t) = Re[V exp(+i omega t)]."""
    _positive(vac_peak=vac_peak)
    return cmath.rect(vac_peak, drive_phase)


def dc_offset(*, vdc: float, v_cpd: float) -> float:
    """U_dc = V_dc - V_cpd: the one DC reference used by every mechanism."""
    return vdc - v_cpd


# ------------------------------------------------------------- field geometry

def point_charge_depth(*, contact_radius: float) -> float:
    """l = a / sqrt(3): point-charge depth matching the on-axis half-potential depth of the top-hat disk."""
    _positive(contact_radius=contact_radius)
    return contact_radius / math.sqrt(3.0)


# ------------------------------------------------------------- tip capacitance

def sphere_capacitance_gradient(*, tip_radius: float, gap: float, eps_r: float) -> float:
    """Apex-only estimate: C_g = (eps_r - 1)/(eps_r + 1) * 2 pi eps0 R^2 / (g (g + R)). F/m, >= 0."""
    _positive(tip_radius=tip_radius, gap=gap, eps_r=eps_r)
    contrast = (eps_r - 1.0) / (eps_r + 1.0)
    return contrast * 2.0 * math.pi * VACUUM_PERMITTIVITY * tip_radius**2 / (gap * (gap + tip_radius))


def capacitance_gradient(*, calibrated: float | None, tip_radius: float, gap: float | None,
                         eps_r: float) -> tuple[float, str] | None:
    """Precedence for C_g: calibrated value, else sphere-plus-gap estimate, else None.

    Returns (C_g [F/m], "calibrated" | "sphere_gap") or None when nothing is
    available; the caller then reports unsupported rather than inventing a gap.
    """
    if calibrated is not None:
        _positive(calibrated=calibrated)
        return calibrated, "calibrated"
    if gap is not None:
        return sphere_capacitance_gradient(tip_radius=tip_radius, gap=gap, eps_r=eps_r), "sphere_gap"
    return None
