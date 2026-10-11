# Conventions and compatibility contract

Every mechanism report names the rows of this page it relies on. Two
mechanisms are compatible, and may be summed by the detector, only when they
share every row.

## Shared choices

| Choice | Value in this repository | Note |
|---|---|---|
| Electrical picture of the tip | a flat disk of radius a, the Hertz contact radius | one length for all mechanisms |
| Point-charge depth for electrostriction and piezo | l = a / sqrt(3) | matches the on-axis half-potential depth of the top-hat boundary used by Vegard; the equipotential-disk far field would give 2a/pi, ten percent larger; piezo at the centre does not depend on l |
| Voltage reaching the sample | eta U(t): eta U_dc and V_j = eta V, same eta for all strain mechanisms | eta from the series layer divider, DC and AC alike; eta = 1 without a layer |
| Sample capacitance scale in the divider | eps_0 eps_r / R_D with mobile ions; eps_0 eps_r / a without | the exact disk gives a factor 4/pi on the second; stated, not applied |
| Voltage driving the electrostatic force | the applied V_ac and U_dc, not V_j | the force acts across the whole junction |
| DC offset | U_dc = V_dc minus V_cpd, for every mechanism | one definition; CPD is a junction-card value, default zero is an assumption |
| Reference bias of the ionic state | U_ref = `ionic.reference_bias_v`; 0 for an untouched card | written by a `material_state` step; only Vegard's bias check uses U_dc minus U_ref, every other use of U_dc is unchanged |
| Ion density in the linear response | N_r = N (1 minus N / N_site), N without a site density | ideal lattice gas, from `scales.responsive_density`; sets R_D for every mechanism and the Vegard prefactor |
| Readout of displacements | disk centre, free surface, outward positive | constant `READOUT_DISPLACEMENT` in `report.py` |
| Readout of forces | net vertical force on the tip apex, outward positive | attraction is negative |
| Elasticity | isotropic, Y and nu from the material card | reductions from tensors happen when the card is built |
| Harmonics | 1w for all; 2w for the two quadratic mechanisms; DC not reported | missing harmonic means not computed |
| Contact law | Hertz | an assumption, not a card field; DMT or JKR would be a later scales option |
| Contact resonance | f_c and Q are measured inputs; k_c is computed | detector reports k_c / k_l |

## Phasors and signs

- Drive: V(t) = V_dc + Re[V_ac e^{i phi} e^{+i omega t}]; V_ac is the peak
  amplitude, phi the drive phase. The complex drive is V = V_ac e^{i phi}.
- Response: x(t) = x_0 + Re[x_1 e^{+i omega t}] + Re[x_2 e^{+2 i omega t}],
  common time origin for both harmonics.
- A quadratic law y = K U(t)^2 gives y_1 = 2 K U_dc V and y_2 = K V^2 / 2.
- z points into the sample; displacement and force are positive outward,
  toward the tip. Ionic charge q = z e keeps its sign everywhere.

## Status words

`valid`, `approximate`, `unsupported`. Anything but `valid` carries reasons.
A computed number is `approximate` whenever a check is outside its range but
the formula is still a stated linear extrapolation. `unsupported` means no
number: missing data, a geometry the formula does not describe, or a check so
far outside range that the formula no longer applies.

## Mechanism-specific rules that keep the four compatible

- **Electromechanical split.** If d is on the card, the piezo node uses it and
  the electrostriction node computes only the induced quadratic part
  (Q chi^2 E^2 plus the Maxwell term), whatever P_0 is. If d is absent and
  P_0 is nonzero, the piezo node derives d from the linearized
  electrostriction (d_33 = 2 chi Q_11 P_0, d_31 = 2 chi Q_12 P_0,
  d_15 = 2 chi (Q_11 minus Q_12) P_0, chi = eps_0 (eps_r minus 1), polar axis
  along z) and says so. The same P_0 term is never counted in both nodes.
- **Vegard under bias.** The formula is a linear response about the card's
  reference state, which is unbiased at U_ref. It is `approximate` while
  |q eta (U_dc minus U_ref)| is at most k_B T and `unsupported` above. Real
  bias effects belong to `material_state/`: a state step absorbs the DC into a
  new card and sets U_ref to the readout DC.
- **State steps.** A `material_state` step runs before the mechanisms. It
  returns a new material card and a `StateReport`. The detector inherits the
  state's status, so a signal is never better than the state it was measured on.
- **Vegard and the layer.** V_j already contains the divider; no second
  division inside the formula.
- **Fields.** The Vegard field is screened by the ions; the electromechanical
  field is the unscreened point-charge field. For a material with both mobile
  ions and d or Q this is a declared inconsistency. Piezoresponse reports it as
  a note. Electrostriction is `approximate`, because its 1w term uses the DC
  field and the ions screen it.
- **Internal Maxwell versus external force.** The electrostriction node is the
  quadratic response of a dielectric to its internal field: electrostriction
  (Q) plus the dielectric Maxwell stress, one field and one quadratic law. The
  Maxwell part is an internal eigenstrain with zero total surface traction.
  The vacuum side of the same field is the external apex force. They are added
  at the detector as displacement and force respectively. This is the P00
  approximation, stated in both reports. The node requires Q and is never
  Maxwell-only. With mobile ions the field does not reach the bulk, and the
  sample's share of the electrostatic loading is the reaction of the external
  force.

## Single-source rule

Every derived scale is computed by a function in `scales.py`. A mechanism may
call it as often as it likes; it may not rewrite it. The contact law and the
capacitance model are card data, so two mechanisms cannot disagree on a scale
unless the cards disagree.
