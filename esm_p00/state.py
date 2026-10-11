"""Graph state for the optional LangGraph assembly in the notebook.

One operating point flows through the graph: the four cards and an optional pulse card
go in; the state report, the mechanism reports and the detector report come out. The
graph itself (nodes, edges, the conditional state step) is built in the notebook,
README rule 7. Nothing else lives here.
"""
from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from esm_p00.cards import ExperimentCard, InstrumentCard, JunctionCard, MaterialCard, PulseCard
from esm_p00.report import DetectorReport, MechanismReport, StateReport


class PointState(TypedDict):
    # inputs
    material: MaterialCard                     # replaced by the prepared card when a state step runs
    experiment: ExperimentCard
    instrument: InstrumentCard
    junction: JunctionCard
    pulse: NotRequired[PulseCard | None]       # None or absent: no state step

    # outputs
    state_report: NotRequired[StateReport]
    reports: NotRequired[Annotated[list[MechanismReport], operator.add]]   # each mechanism node appends one
    detector: NotRequired[DetectorReport]
