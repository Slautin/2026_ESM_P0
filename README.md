# 2026_ESM_P0

Minimal forward model for electrochemical strain microscopy (ESM): four
independent analytical mechanisms and one linear detector, assembled in a
notebook. This is the "minimal lovable product" agreed on 6 October 2026:
linear, small-signal, additive, algebraic.

## Vocabulary

- **P00** (this repository): the closed-form small-signal response of one
  mechanism at the disk centre under a prescribed junction voltage, with every
  mechanism independent and all of them sharing the conventions in
  `docs/CONVENTIONS.md`. Mechanisms are combined only by the linear detector.
- **State + response** (`material_state/`): a DC bias or pulse first
  changes the material card; the P00 formulas are then evaluated around it.
- **P1** (not here): differential equations in time or space.

The older repository `2026_ESM_oracle` uses "P0" for a static planar
boundary-value branch and "P00" for its independent analytical layer. This
repository takes the formulas of that P00 layer and nothing else.

## Layout

```
esm_p00/
  constants.py        e, k_B, eps_0
  cards.py            Param plus MaterialCard, ExperimentCard, InstrumentCard, JunctionCard, PulseCard;
                      card.draw(rng) samples the uncertain values, card.draw(mean=True) their means,
                      card.at(z) places them at prior-width coordinates
  scales.py           pure functions: Hertz contact, Debye scales, voltage divider,
                      junction voltage, point-charge depth, sphere capacitance gradient
  report.py           Check, MechanismReport, StateReport, DetectorReport; harmonic keys, status order,
                      readout strings
  mechanisms/         vegard, piezoresponse, electrostriction, electrostatic
  detector.py         two springs, contact-resonance oscillator, detector factor
  protocols.py        Step, SingleFrequency, FrequencySweep, PulseReadout: experiments as lists of
                      operating points
  particles.py        resample-move particle filter over the uncertain card values; updated cards
  examples.py         demo cards with sources: LiCoO2, illustrative polar dielectric, AFM probe,
                      ambient junction (residual CPD, water layer)
  state.py            PointState: the TypedDict for the optional LangGraph graph
material_state/       DC state steps run before the mechanisms: occupancy (lattice gas,
                      diffusive recovery after a pulse)
notebooks/            01_forward_p00.ipynb: the LangGraph graph, a frequency sweep, a pulse relaxation,
                      linear and quadratic channels
                      02_forward_uncertainty.ipynb: Monte Carlo bands with an uncertain interface,
                      which parameter matters where
                      03_backward_particle_filter.ipynb: a synthetic truth, a sweep, the filter,
                      updated prediction and parameter distributions
docs/                 PHYSICS.md (formulas), CONVENTIONS.md (compatibility table)
tests/                one file per module: a known limit and a sign check
```

## Rules

1. No mechanism imports another mechanism.
2. Every derived scale (contact radius, stiffness, Debye length, voltage
   fraction, point-charge depth, capacitance gradient) comes from `scales.py`.
   Mechanisms recompute them; they never copy the formula.
3. Each mechanism file has three blocks: checks, calculation, output. Its
   required inputs are listed at the top. Its physics is a pure function; the
   graph node is a two-line wrapper.
4. Missing data is unavailable, never zero. A result outside its checks is
   reported as approximate with reasons, or withheld as unsupported.
5. Every report lists the scales it actually used, so the notebook can show
   that all mechanisms agree on them.
6. The detector refuses displacements whose readout statement differs.
7. No module combines mechanisms. The notebook wires them; the detector sums
   them by linearity.

## Status

Cards, scales, reports, the four mechanisms, the detector, the occupancy state step and the protocols
(single frequency, frequency sweep, pulse readout) are implemented, all closed-form, together with the
LangGraph state (`PointState`), the demo cards, forward Monte Carlo uncertainty, the backward particle
filter and the three notebooks. Only the import smoke test exists; the per-module tests are placeholders.
Physics and conventions are written in `docs/`. See `ORACLE_P00_REVIEW.md` for the context and the
decisions taken.

## Run

```bash
.venv/bin/python -m pytest -q
```
