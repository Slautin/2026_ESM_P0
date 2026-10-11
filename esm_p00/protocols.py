"""Protocols: an experiment described as a list of operating points.

A protocol holds the readout and what is varied, and expands into Steps. A Step is one
operating point: one ExperimentCard and, optionally, one PulseCard. The physics evaluates
one Step at a time: the state step if there is a pulse, then the mechanisms and the
detector. The notebook loops over the Steps. There is no physics here, and nothing here
is imported by a mechanism, a state step or the detector.

A new waveform is a new class with a steps() method. Planned: AC sweep, on-field DC sweep,
and loops (the first to need a state carried from one step to the next).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from esm_p00.cards import ExperimentCard, PulseCard

_CFG = ConfigDict(extra="forbid", frozen=True)


class Step(BaseModel):
    """One operating point: what the physics evaluates once."""

    model_config = _CFG

    x: float                          # the swept value; its meaning is in the protocol's docstring
    experiment: ExperimentCard
    pulse: PulseCard | None = None    # None: no state step


class SingleFrequency(BaseModel):
    """One AC readout at one frequency. x is the frequency in Hz.

    Needs: the mechanisms and the detector.
    """

    model_config = _CFG

    readout: ExperimentCard

    def steps(self) -> list[Step]:
        return [Step(x=self.readout.frequency_hz, experiment=self.readout)]


class FrequencySweep(BaseModel):
    """The same AC readout at each frequency. x is the frequency in Hz.

    Needs: the mechanisms and the detector.
    """

    model_config = _CFG

    readout: ExperimentCard          # its frequency_hz is replaced by each value of frequencies_hz
    frequencies_hz: list[float]

    def steps(self) -> list[Step]:
        return [Step(x=f, experiment=ExperimentCard(**{**self.readout.model_dump(), "frequency_hz": f}))
                for f in self.frequencies_hz]


class PulseReadout(BaseModel):
    """A DC pulse, then an AC readout at each delay. x is the delay in s.

    Needs: the occupancy state step; a readout much faster than the recovery.
    """

    model_config = _CFG

    readout: ExperimentCard          # its vdc_v is the readout DC (0 for off-field)
    pulse: PulseCard                 # its delay_s is replaced by each value of delays_s
    delays_s: list[float]

    def steps(self) -> list[Step]:
        return [Step(x=t, experiment=self.readout, pulse=PulseCard(**{**self.pulse.model_dump(), "delay_s": t}))
                for t in self.delays_s]
