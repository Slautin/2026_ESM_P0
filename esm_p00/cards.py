"""The input cards: plain SI numbers in the form the formulas consume.

Material, experiment, instrument and junction cards feed every mechanism; the
optional pulse card feeds only the material_state steps.

Every physical number is a `Param`: a nominal `value` plus an optional
marginal distribution centred on it. Formulas read only `.value`; the
distribution is carried for later propagation and inference. Provenance is
not stored here; it belongs to the stage that builds the card.

Cards are frozen and refuse unknown fields. Optional blocks that are absent
mean "data not available", which makes the mechanism needing them report
unsupported. Absence is never a zero.

A bare number is accepted wherever a Param is expected, so a card can be
written as `eps_r=10.0` or as `eps_r=Param(value=10.0, dist="normal",
dist_params={"sigma": 3.0})`.
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

_CFG = ConfigDict(extra="forbid", frozen=True)

Distribution = Literal["fixed", "normal", "lognormal", "uniform"]

# Keys that each distribution needs. All distributions are centred on `value`:
# the mean for normal, the median for lognormal, the midpoint for uniform.
_DIST_KEYS: dict[str, frozenset[str]] = {
    "fixed": frozenset(),
    "normal": frozenset({"sigma"}),
    "lognormal": frozenset({"sigma_ln"}),
    "uniform": frozenset({"half_width"}),
}


class Param(BaseModel):
    """One physical number with an optional marginal distribution."""

    model_config = _CFG

    value: float
    dist: Distribution = "fixed"
    dist_params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_number(cls, data):
        if isinstance(data, bool):
            raise ValueError("a Param value must be a number, not a boolean")
        if isinstance(data, (int, float)):
            return {"value": float(data)}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> "Param":
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        expected = _DIST_KEYS[self.dist]
        given = frozenset(self.dist_params)
        if given != expected:
            raise ValueError(
                f"dist '{self.dist}' needs parameters {sorted(expected)}, got {sorted(given)}"
            )
        for key, width in self.dist_params.items():
            if not math.isfinite(width) or width <= 0.0:
                raise ValueError(f"dist parameter '{key}' must be finite and > 0")
        if self.dist == "lognormal" and self.value <= 0.0:
            raise ValueError("a lognormal Param needs a positive value")
        return self


# ---------------------------------------------------------------- constraints

def _bounded(low: float | None, high: float | None, message: str) -> AfterValidator:
    def _check(p: Param) -> Param:
        if (low is not None and p.value <= low) or (high is not None and p.value >= high):
            raise ValueError(message)
        return p
    return AfterValidator(_check)


Positive = Annotated[Param, _bounded(0.0, None, "value must be > 0")]
Permittivity = Annotated[Param, _bounded(1.0 - 1e-12, None, "eps_r must be >= 1")]
PoissonRatio = Annotated[Param, _bounded(-1.0, 0.5, "poisson_ratio must lie in (-1, 0.5)")]


def _finite(name: str):
    return Field(allow_inf_nan=False, description=name)


# ------------------------------------------------------------------ material

class IonicBlock(BaseModel):
    """One dominant mobile ionic species. Present only when data exist."""

    model_config = _CFG

    charge_number: int                                   # signed, nonzero
    number_density: Positive                             # m^-3, mobile ions
    site_density: Positive | None = None                 # m^-3, available sites
    diffusivity: Positive                                # m^2/s
    diffusivity_kind: Literal["self", "chemical", "unknown"] = "unknown"
    vegard_11: Param                                     # m^3 per ion, in-plane
    vegard_33: Param                                     # m^3 per ion, along z

    # Junction DC offset U_dc at which this ionic state is the unbiased reference.
    # 0 for an untouched card; written by material_state when it prepares a state.
    reference_bias_v: float = Field(default=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _physical(self) -> "IonicBlock":
        if self.charge_number == 0:
            raise ValueError("charge_number must be nonzero")
        if self.site_density is not None and self.number_density.value >= self.site_density.value:
            raise ValueError("number_density must be below site_density: a full lattice has no mobile ions")
        return self


class PiezoBlock(BaseModel):
    """Bulk incremental piezoelectric coefficients, polar axis along z.

    d_15 uses the engineering convention (twice the full-tensor shear entry).
    """

    model_config = _CFG

    d_31: Param                                          # m/V
    d_33: Param                                          # m/V
    d_15: Param                                          # m/V


class ElectrostrictionBlock(BaseModel):
    """Isotropic strain-based electrostriction and spontaneous polarization.

    Q is the bare strain coefficient for the total polarization; it excludes
    Maxwell loading. P_0 is the component along z; zero for a paraelectric.
    """

    model_config = _CFG

    Q_11: Param                                          # m^4/C^2
    Q_12: Param                                          # m^4/C^2
    P_0: Param                                           # C/m^2


class MaterialCard(BaseModel):
    """What the material contributes. Required: dielectric and elastic data."""

    model_config = _CFG

    name: str = Field(min_length=1)
    note: str = ""

    eps_r: Permittivity                                  # relative, isotropic
    youngs_modulus: Positive                             # Pa
    poisson_ratio: PoissonRatio                          # dimensionless

    ionic: IonicBlock | None = None
    piezo: PiezoBlock | None = None
    electrostriction: ElectrostrictionBlock | None = None

    @property
    def has_mobile_ions(self) -> bool:
        return self.ionic is not None


# ---------------------------------------------------------------- experiment

class ExperimentCard(BaseModel):
    """Control settings of one sinusoidal measurement. Plain numbers."""

    model_config = _CFG

    vac_peak_v: float = Field(gt=0.0, allow_inf_nan=False)       # peak, not RMS
    vdc_v: float = Field(default=0.0, allow_inf_nan=False)       # as set on the instrument
    frequency_hz: float = Field(gt=0.0, allow_inf_nan=False)
    drive_phase_rad: float = Field(default=0.0, allow_inf_nan=False)
    temperature_k: float = Field(gt=0.0, allow_inf_nan=False)
    normal_load_n: float = Field(gt=0.0, allow_inf_nan=False)    # Hertz needs a load

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.frequency_hz


class PulseCard(BaseModel):
    """DC history before one readout: a pulse, then a delay. Plain numbers. Optional.

    Read by a material_state step, never by a mechanism. On-field readout is
    delay_s = 0 with the ExperimentCard vdc_v equal to pulse_v; off-field readout
    is a delay with the readout DC usually at 0. Protocols set delay_s per step.
    """

    model_config = _CFG

    pulse_v: float = Field(allow_inf_nan=False)                  # as set on the instrument, like vdc_v
    pulse_s: float = Field(gt=0.0, allow_inf_nan=False)          # pulse duration
    delay_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)   # end of pulse to readout
    readout_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)   # None: frozen readout assumed


# ---------------------------------------------------------------- instrument

class InstrumentCard(BaseModel):
    """Tip, cantilever and detection. Calibration multipliers default to 1."""

    model_config = _CFG

    tip_radius_m: Positive
    tip_young_modulus_pa: Positive
    tip_poisson_ratio: PoissonRatio
    cantilever_stiffness_n_m: Positive

    # Measured in contact; both or neither. Neither means quasi-static H = 1.
    contact_resonance_hz: Positive | None = None
    contact_quality_factor: Positive | None = None

    # Measured dC/dz at the operating point; replaces the sphere estimate.
    capacitance_gradient_f_m: Positive | None = None

    # 1.0 means "not calibrated" and is reported as such.
    detector_factor: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    force_efficiency_factor: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _resonance_pair(self) -> "InstrumentCard":
        if (self.contact_resonance_hz is None) != (self.contact_quality_factor is None):
            raise ValueError("contact_resonance_hz and contact_quality_factor must be given together")
        return self

    @property
    def has_contact_resonance(self) -> bool:
        return self.contact_resonance_hz is not None


# ------------------------------------------------------------------ junction

class JunctionCard(BaseModel):
    """What sits between tip and sample. Defaults are assumptions, not data."""

    model_config = _CFG

    v_cpd_v: Param = Field(default_factory=lambda: Param(value=0.0))

    # Optional dielectric layer in series with the sample; both or neither.
    surface_layer_thickness_m: Positive | None = None
    surface_layer_relative_permittivity: Positive | None = None

    # Vacuum gap for the sphere capacitance estimate. Never inferred from the layer:
    # the apex formula diverges as 1/g, so a layer thickness is not a usable gap.
    gap_m: Positive | None = None

    @model_validator(mode="after")
    def _layer_pair(self) -> "JunctionCard":
        if (self.surface_layer_thickness_m is None) != (self.surface_layer_relative_permittivity is None):
            raise ValueError("surface layer thickness and permittivity must be given together")
        return self

    @property
    def has_surface_layer(self) -> bool:
        return self.surface_layer_thickness_m is not None
