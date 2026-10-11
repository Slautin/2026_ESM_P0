"""The output types: one per mechanism, one per state step, one for the detector.

A report is the only object that crosses a mechanism's boundary. Its
validators enforce the rules that are easy to break by accident:

- a missing harmonic means "not computed", never zero;
- an unsupported result carries no numbers;
- status is derived from the named checks and can be forced worse with a
  reason, never better;
- every status other than valid carries reasons;
- a displacement is always read at the disk centre, a force always at the
  apex, so the detector may sum them.

Phasors follow x(t) = x0 + Re[x_h exp(+i h omega t)] with peak amplitudes
and a common time origin. The compatibility table these strings implement is
docs/CONVENTIONS.md.
"""
from __future__ import annotations

import cmath
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_CFG = ConfigDict(extra="forbid", frozen=True)

HARMONICS = ("1w", "2w")
Harmonic = Literal["1w", "2w"]

STATUS = ("valid", "approximate", "unsupported")      # in order of badness
Status = Literal["valid", "approximate", "unsupported"]

Observable = Literal["displacement", "force"]
READOUT_DISPLACEMENT = "disk centre, free surface, outward positive"
READOUT_FORCE = "net vertical force on the tip apex, outward positive"
READOUT = {"displacement": READOUT_DISPLACEMENT, "force": READOUT_FORCE}
UNIT = {"displacement": "m", "force": "N"}


def worst(*statuses: Status) -> Status:
    """The worse of several statuses; valid when none are given."""
    return max(statuses, key=STATUS.index) if statuses else "valid"


def _finite_phasors(values: dict) -> dict:
    for key, z in values.items():
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise ValueError(f"harmonic {key} must be finite")
    return values


# ------------------------------------------------------------------- checks

class Check(BaseModel):
    """One applicability number, its limit, and what failing it means."""

    model_config = _CFG

    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    limit: float | None = None                     # None: no numeric limit
    ok: bool = True
    on_fail: Literal["informational", "approximate", "unsupported"] = "approximate"
    note: str = ""                                 # reason sentence for flags

    @classmethod
    def upper(cls, name: str, value: float, limit: float,
              on_fail: Literal["approximate", "unsupported"] = "approximate") -> "Check":
        """Passes when value <= limit."""
        return cls(name=name, value=value, limit=limit, ok=value <= limit, on_fail=on_fail)

    @classmethod
    def lower(cls, name: str, value: float, limit: float,
              on_fail: Literal["approximate", "unsupported"] = "approximate") -> "Check":
        """Passes when value >= limit."""
        return cls(name=name, value=value, limit=limit, ok=value >= limit, on_fail=on_fail)

    @classmethod
    def info(cls, name: str, value: float) -> "Check":
        """A number worth reporting that has no limit."""
        return cls(name=name, value=value, on_fail="informational")

    @classmethod
    def flag(cls, name: str, ok: bool, note: str,
             on_fail: Literal["informational", "approximate", "unsupported"] = "informational") -> "Check":
        """A yes-or-no model-quality statement; informational by default."""
        return cls(name=name, value=1.0 if ok else 0.0, ok=ok, on_fail=on_fail, note=note)

    @property
    def affects_status(self) -> bool:
        return not self.ok and self.on_fail != "informational"

    @property
    def reason(self) -> str:
        if self.note:
            return self.note
        if self.limit is None:
            return f"{self.name} = {self.value:.3g} failed"
        side = "above" if self.value > self.limit else "below"
        return f"{self.name} = {self.value:.3g} is {side} its limit {self.limit:.3g}"


def status_from_checks(checks: list[Check]) -> Status:
    return worst(*(c.on_fail for c in checks if c.affects_status))


# ------------------------------------------------------------------- common

class _Report(BaseModel):
    model_config = _CFG

    checks: list[Check] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    scales_used: dict[str, float] = Field(default_factory=dict)
    status: Status = "valid"
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_rules(self) -> "_Report":
        derived = status_from_checks(self.checks)
        if STATUS.index(self.status) < STATUS.index(derived):
            raise ValueError(f"status '{self.status}' is better than the checks allow ('{derived}')")
        if self.status != "valid" and not any(r.strip() for r in self.reasons):
            raise ValueError(f"status '{self.status}' requires at least one reason")
        if any(not a.strip() for a in self.assumptions) or any(not r.strip() for r in self.reasons):
            raise ValueError("assumptions and reasons must be nonblank sentences")
        for name, value in self.scales_used.items():
            if not math.isfinite(value):
                raise ValueError(f"scale {name} must be finite")
        return self

    @property
    def is_supported(self) -> bool:
        return self.status != "unsupported"

    @property
    def failed_checks(self) -> list[Check]:
        """Failed checks that affect the status."""
        return [c for c in self.checks if c.affects_status]

    @property
    def notes(self) -> list[str]:
        """Failed informational flags: worth knowing, not a downgrade."""
        return [c.reason for c in self.checks if not c.ok and c.on_fail == "informational"]


def _resolve(checks: list[Check], reasons: list[str], force_unsupported: bool) -> tuple[Status, list[str]]:
    """Status from the checks, optionally forced to unsupported; reasons for every failed check."""
    status = status_from_checks(checks)
    if force_unsupported:
        status = "unsupported"
    return status, [c.reason for c in checks if c.affects_status] + list(reasons)


# ---------------------------------------------------------------- mechanism

class MechanismReport(_Report):
    """One mechanism's observable, applicability and assumptions."""

    mechanism: str = Field(min_length=1)
    observable: Observable
    readout: str | None = None                    # filled from the observable
    harmonics: dict[Harmonic, complex] = Field(default_factory=dict)
    reference: str = ""

    @model_validator(mode="after")
    def _mechanism_rules(self) -> "MechanismReport":
        expected = READOUT[self.observable]
        if self.readout is None:
            object.__setattr__(self, "readout", expected)
        elif self.readout != expected:
            raise ValueError(f"a {self.observable} must be read out as '{expected}'")
        _finite_phasors(self.harmonics)
        if self.is_supported:
            if not self.harmonics:
                raise ValueError("a supported mechanism must supply at least one harmonic; missing is not zero")
            if not self.assumptions:
                raise ValueError("a supported mechanism must state its assumptions")
        elif self.harmonics:
            raise ValueError("an unsupported mechanism must not supply numbers")
        return self

    @classmethod
    def build(cls, *, mechanism: str, observable: Observable, harmonics: dict[str, complex],
              checks: list[Check], assumptions: list[str], scales_used: dict[str, float],
              reasons: list[str] = (), reference: str = "") -> "MechanismReport":
        """Supported result: status and reasons follow from the checks."""
        status, all_reasons = _resolve(checks, list(reasons), force_unsupported=False)
        return cls(mechanism=mechanism, observable=observable, harmonics=harmonics, checks=checks,
                   assumptions=assumptions, scales_used=scales_used, status=status,
                   reasons=all_reasons, reference=reference)

    @classmethod
    def unsupported(cls, *, mechanism: str, observable: Observable, reasons: list[str],
                    checks: list[Check] = (), assumptions: list[str] = (),
                    scales_used: dict[str, float] | None = None, reference: str = "") -> "MechanismReport":
        """No number: missing data, wrong geometry, or a check far outside its range."""
        _, all_reasons = _resolve(list(checks), list(reasons), force_unsupported=True)
        return cls(mechanism=mechanism, observable=observable, checks=list(checks),
                   assumptions=list(assumptions), scales_used=scales_used or {},
                   status="unsupported", reasons=all_reasons, reference=reference)

    @property
    def unit(self) -> str:
        return UNIT[self.observable]

    def amplitude(self, harmonic: Harmonic) -> float | None:
        z = self.harmonics.get(harmonic)
        return None if z is None else abs(z)

    def phase_deg(self, harmonic: Harmonic) -> float | None:
        z = self.harmonics.get(harmonic)
        return None if z is None or z == 0 else math.degrees(cmath.phase(z))   # phase of zero is undefined

    def row(self) -> dict[str, object]:
        """Flat summary for a notebook table."""
        out: dict[str, object] = {"mechanism": self.mechanism, "observable": self.observable,
                                  "unit": self.unit, "status": self.status}
        for h in HARMONICS:
            out[f"|{h}|"] = self.amplitude(h)
            out[f"phase {h} (deg)"] = self.phase_deg(h)
        out["failed checks"] = ", ".join(c.name for c in self.failed_checks)
        out["notes"] = "; ".join(self.notes)
        return out


# ----------------------------------------------------------------- detector

class DetectorReport(_Report):
    """Measured signal per mechanism and in total, after the instrument transfer."""

    per_mechanism: dict[str, dict[Harmonic, complex]] = Field(default_factory=dict)
    total: dict[Harmonic, complex] = Field(default_factory=dict)
    transfer: dict[Harmonic, complex] = Field(default_factory=dict)
    omitted: list[str] = Field(default_factory=list)   # unsupported mechanisms, not summed

    @model_validator(mode="after")
    def _detector_rules(self) -> "DetectorReport":
        for values in (*self.per_mechanism.values(), self.total, self.transfer):
            _finite_phasors(values)
        if not self.is_supported:
            if self.per_mechanism or self.total:
                raise ValueError("an unsupported detector result must not supply numbers")
            return self
        expected = {h: sum((v.get(h, 0j) for v in self.per_mechanism.values()), 0j)
                    for h in HARMONICS if any(h in v for v in self.per_mechanism.values())}
        if set(expected) != set(self.total):
            raise ValueError("total must contain exactly the harmonics present per mechanism")
        for h, z in expected.items():
            scale = max(abs(z), max(abs(v.get(h, 0j)) for v in self.per_mechanism.values()), 1e-300)
            if abs(self.total[h] - z) > 1e-9 * scale:
                raise ValueError(f"total at {h} is not the complex sum of the contributions")
        return self

    @classmethod
    def build(cls, *, per_mechanism: dict[str, dict[str, complex]], transfer: dict[str, complex],
              checks: list[Check], assumptions: list[str], scales_used: dict[str, float],
              omitted: list[str] = (), reasons: list[str] = ()) -> "DetectorReport":
        total = {h: sum((v.get(h, 0j) for v in per_mechanism.values()), 0j)
                 for h in HARMONICS if any(h in v for v in per_mechanism.values())}
        status, all_reasons = _resolve(checks, list(reasons), force_unsupported=False)
        return cls(per_mechanism=per_mechanism, total=total, transfer=transfer, checks=checks,
                   assumptions=assumptions, scales_used=scales_used, omitted=list(omitted),
                   status=status, reasons=all_reasons)

    @classmethod
    def unsupported(cls, *, reasons: list[str], checks: list[Check] = (), assumptions: list[str] = (),
                    scales_used: dict[str, float] | None = None, omitted: list[str] = ()) -> "DetectorReport":
        """No numbers: nothing to sum, or contributions that may not be summed."""
        _, all_reasons = _resolve(list(checks), list(reasons), force_unsupported=True)
        return cls(checks=list(checks), assumptions=list(assumptions), scales_used=scales_used or {},
                   omitted=list(omitted), status="unsupported", reasons=all_reasons)

    def amplitude(self, harmonic: Harmonic) -> float | None:
        z = self.total.get(harmonic)
        return None if z is None else abs(z)

    def phase_deg(self, harmonic: Harmonic) -> float | None:
        z = self.total.get(harmonic)
        return None if z is None or z == 0 else math.degrees(cmath.phase(z))

    def row(self) -> dict[str, object]:
        out: dict[str, object] = {"status": self.status}
        for h in HARMONICS:
            out[f"|{h}|"] = self.amplitude(h)
            out[f"phase {h} (deg)"] = self.phase_deg(h)
        out["omitted"] = ", ".join(self.omitted)
        out["failed checks"] = ", ".join(c.name for c in self.failed_checks)
        return out


# -------------------------------------------------------------------- state

class StateReport(_Report):
    """What a state step did to the material card, and how far to trust it."""

    step: str = Field(min_length=1)
    values: dict[str, float] = Field(default_factory=dict)   # state results, e.g. theta_0, theta

    @model_validator(mode="after")
    def _state_rules(self) -> "StateReport":
        for name, value in self.values.items():
            if not math.isfinite(value):
                raise ValueError(f"state value {name} must be finite")
        if self.is_supported:
            if not self.values:
                raise ValueError("a supported state must supply its values")
            if not self.assumptions:
                raise ValueError("a supported state must state its assumptions")
        elif self.values:
            raise ValueError("an unsupported state must not supply numbers")
        return self

    @classmethod
    def build(cls, *, step: str, values: dict[str, float], checks: list[Check], assumptions: list[str],
              scales_used: dict[str, float], reasons: list[str] = ()) -> "StateReport":
        """Supported result: status and reasons follow from the checks."""
        status, all_reasons = _resolve(checks, list(reasons), force_unsupported=False)
        return cls(step=step, values=values, checks=checks, assumptions=assumptions,
                   scales_used=scales_used, status=status, reasons=all_reasons)

    @classmethod
    def unsupported(cls, *, step: str, reasons: list[str], checks: list[Check] = (),
                    assumptions: list[str] = (), scales_used: dict[str, float] | None = None) -> "StateReport":
        """No new state: the card is passed on unchanged and the detector refuses to sum."""
        _, all_reasons = _resolve(list(checks), list(reasons), force_unsupported=True)
        return cls(step=step, checks=list(checks), assumptions=list(assumptions),
                   scales_used=scales_used or {}, status="unsupported", reasons=all_reasons)
