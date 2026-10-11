# material_state

DC-driven changes of the material under the tip. A state step evaluates the P00
response formulas around a modified material card, instead of extending those
formulas with bias terms.

Contract for every step:

- `compute(material, experiment, instrument, junction, <step card>)` returns
  `(new MaterialCard, StateReport)`. On unsupported, the original card is
  returned unchanged and the report carries no numbers.
- It uses the shared scales from `esm_p00/scales.py`, so a, eta and R_D are
  the same as in the mechanisms.
- It writes `ionic.reference_bias_v`, so Vegard does not count the absorbed DC
  twice.
- The graph wrapper `node(state)` returns `{"material": new_card, "state_report": report}`.
  The mechanism nodes after it see the new card, and the detector inherits the
  report's status.
- Nothing in `esm_p00` imports this folder.

Steps:

- `occupancy.py`: lattice-gas occupancy under the tip after a DC pulse, then
  diffusive recovery for off-field readout (docs/PHYSICS.md section 8). It
  reads a `PulseCard` from `esm_p00.cards` with `pulse_v`, `pulse_s`, `delay_s`
  and `readout_s` (optional).

Later candidates: bias-induced polarization that turns electrostriction into an
effective piezoelectric coefficient; a shifted contact potential after charge
injection; pulse trains with memory.
