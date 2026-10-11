# Context review for 2026_ESM_P0

Written 2026-10-10 after reading: the 6 Oct 2026 meeting transcript (ESM PKM.docx),
the PKM schematic (ESM_PKM_Schematic.pdf), the consolidated physics note
`P00_PHYSICS.md` from the Codex project, the whole `2026_ESM_oracle` repository
(code, docs, notebooks, tests) and the Codex patch bundles P00 Patch 1-4.
No code is proposed here. This is the shared understanding to structure the new
small project against.

---

## 1. What the meeting fixed as the target

Participants: Sergei Kalinin, Boris Slautin, Faith Hines. Decisions that matter for
the new repository:

1. **Two arrows.** Source knowledge (papers, data) -> reconciled material card
   (rigid marginal distributions per parameter, joint distributions later) ->
   physics engine -> observables, and later backward from observations to
   posteriors on the card. The backward arrow to source evidence ("these papers
   are wrong") is deferred.
2. **Separate the material PKM from the instrument model as far as physics
   allows.** The material PKM returns material responses (surface displacement vs
   frequency and voltage, surface potential, charge, domain size) that depend on
   as few instrument parameters as possible. The instrument model (contact,
   cantilever transfer function, detector) turns them into measured signals.
   Contact radius is the acknowledged coupling that cannot be removed.
3. **Voltage waveform is a control input, not an instrument parameter.** The
   action space (sine, pulse, pulse + readout, hysteresis-loop waveform,
   arbitrary) is fixed by the instrument; for each waveform choose the simplest
   material and instrument model that can still calculate the response.
4. **Complexity ladder.** Purely algebraic / analytical formulas first; promote to
   ODE or PDE only when the experiment (large amplitude, coupled signals, pulse
   plus AC) or the physics (an applicability limit is hit) demands it. Simpler
   models must be the small-signal, no-coupling limit of the complex ones.
5. **Composability.** A pulse or DC offset changes the material state; the small
   AC response is then evaluated around that state. That split (state model +
   response model) is the next level after the purely algebraic one.
6. **Every mechanism carries explicit applicability.** Human-interpretable at all
   levels: "Debye approximation valid / not valid" rather than hidden judgement.
   The engine may answer "I do what you ask, but you are outside the P0 limits."
7. **Minimal lovable product (Boris's words, confirmed by Sergei):**
   - material PKM = Morozovska formula for the ionic (Vegard) response plus a
     capacitor-like electrostatic term; linear, small-signal, additive,
     algebraic; nothing else;
   - instrument = a simple transfer function (damped oscillator / contact
     resonance; Euler-Bernoulli later if a broad frequency response is needed);
   - experiment planning = the list of waveforms, with a clear statement of
     what the material and instrument models must satisfy for each;
   - one or two worked demonstrations: build the card, predict the frequency
     response, show the forward path, and prepare the backward path;
   - the first paper is about the workflow; the demonstration is a small part.
8. **Vocabulary is unsettled.** In the meeting Sergei proposed: P0 = purely
   algebraic; "P00" = split into material-state model plus response model;
   P1 = differential equations. In the oracle repository "P00" already means the
   independent analytical layer and "P0" means the static planar branch. The new
   repository should fix one vocabulary in its README on day one.
9. **Delegation.** Faith builds the literature extraction (equations and boundary
   conditions as the primary keys, parameters second); Boris owns the data
   structures (material card, instrument digital twin) and the PKM itself.

## 2. What the schematic adds

The PDF is the block diagram of item 1: Source knowledge -> Reconciled parameters
and initial-state priors p(theta, x0) -> Material physics engine (analytical,
ODE, PDE; "minimum adequate model"; state preparation + readout; state variables
passed explicitly) -> Physical response (displacement, surface potential, charge,
impedance, geometry) -> Instrument PKM (excitation and contact; mechanical
transfer; detection with gain, phase, noise; calibration constraints) -> Signals
and comparison (predicted vs measured, likelihood, inference). The experimental
protocol (waveform, timing, acquisition) feeds both the material engine and the
instrument model. The illustrative panel is the one-line summary of the MLP:
`Y(omega) = R(omega) x H(omega)` for a single linear channel.

## 3. What the oracle repository is today

Repository: `/Users/borisslautin/Documents/GitHub/2026_ESM_oracle`, branch main,
19 commits, head `8f8acd5 p00` (2026-10-10), working tree clean. About 7100
lines in `esm_oracle/`, 50 test files; the full suite passes today (633 tests,
about two minutes, run from the repository's own venv). Python 3.11 venv with
numpy 2.4, scipy 1.17, pydantic 2.13, langgraph.

Three execution routes coexist behind one input object, `ForwardInputs`:

| Route | Where | What it computes | Fidelity |
|---|---|---|---|
| Spatial harmonic graph (Steps 1-8, Sep 2026) | `graph/`, `physics/electro_ionic.py`, `elastic.py`, `detection.py`, `mechanisms/` | Hankel-domain chain: contact, screening, tip capacitance, coupled Poisson-Nernst-Planck with blocking surface (dielectric / linear PNP / saturated-layer branches), Joule heating, four mechanisms (Vegard, electromechanical, thermal, electrostatic), Boussinesq half-space elastic solver, two-springs detection with contact resonance | numerical fields on (k, z) grids |
| "P0" static planar branch (Oct 2026) | `physics/equilibrium*.py`, `inputs/p0.py`, notebook 08 | Closed ion-blocking planar cell, lattice-gas equilibrium as a 1D boundary-value problem, constrained voltage tangent, planar clamped-film strain readout, separate interface force | numerical 1D BVP, DC and per-volt only, no frequency |
| P00 independent analytical layer (Patches 1-3 installed, Patch 4 prepared) | `physics/p00/` | Per-mechanism closed-form or semi-analytical 1w observables, each with its own validity page | algebra plus one scalar quadrature |

The oracle's design rules, stated in `ARCHITECTURE.md`, the two reports and the
patch notes, and visible in the code:

- Every physical number is a `Quantity` or `TensorQuantity` with unit (refused,
  never converted), 1-sigma uncertainty, sources, conditions, resolution method,
  confidence and warnings. Full tensors in the lab frame; Voigt only at ingest.
- Every stage returns the same envelope: `ModelInfo` (name, fidelity,
  assumptions), `Validity` (valid / approximate / unsupported plus reasons),
  dimensionless regime numbers, diagnostics, provenance. Worst status wins.
- Peak phasors, `x(t) = x0 + Re[x e^{+i omega t}]`, V_ac is peak, outward
  displacement and outward force are positive, z into the sample.
- Contributions add as complex numbers, never as amplitudes; per-mechanism
  breakdown is preserved.
- Missing data are unavailable, not zero; an explicit zero coefficient gives an
  explicit zero response. Failures return unsupported with reasons; programming
  errors propagate.
- Models are data (`ModelChoices`), so switching a formula is a one-word change
  recorded in the input fingerprint.

## 4. The P00 realization in detail

### 4.1 Contracts (Patch 1, `physics/p00/common.py`, `run.py`)

- `P00HarmonicResponse`: unit (m or N), a mandatory `reference` string naming
  the readout point or force convention, complex values per harmonic (1w, 2w),
  plus the Quantity provenance fields.
- `P00MechanismResult` extends `StageResult`: supported results must declare
  assumptions and supply at least one observable; unsupported results carry
  reasons and no numbers. Observables are `surface_displacement`,
  `tip_displacement`, `tip_force`.
- `P00Prediction`: inputs plus a dict of named contributions. Aggregate views
  (`overall_validity`, `assumptions`, `warnings`, `breakdown`) are derived
  properties. No sum, no compatibility claim, no force-to-displacement.
- `run_p00(inputs, mechanisms={name: callable})`: executes only explicitly
  registered callbacks, in order. Empty registry means unsupported. No global
  registry, no discovery from material data.

### 4.2 Ionic scalar preparation (Patch 2, `prepare.py`)

`prepare_ionic(inputs)` reuses the existing contact helper (Hertz/DMT/JKR with
the indentation modulus) and the dilute Debye formula and returns a, R_D,
omega_M = D/R_D^2, the layer voltage fraction eta (static planar series
capacitor) and the complex effective AC voltage, together with regime numbers
a/R_D, |qV|/kT, omega/omega_M, omega a^2/D, R_D N^(1/3) and a long list of
cautions (large drive, non-dilute, diffusivity kind, DC/CPD supplied, layer at
high frequency, temperature mismatch).

### 4.3 Vegard response (Patch 3, `vegard.py`)

`compute(inputs)` evaluates Morozovska, Eliseev and Kalinin (2012) Eq. 9a at the
disk centre: outward displacement at 1w equals minus q N a V_eff / kT times a
dimensionless Bessel integral in which S = sqrt(x^2 + alpha^2 + i Omega) and the
elastic kernel B(x) carries beta_11, beta_33 and nu. Numerical work is one scalar
adaptive quadrature with an exact Fourier tail and a repeat check; the result
carries a numerical error estimate. Refusals: nonzero DC or CPD (no biased
equilibrium), finite P0 blocking cell, missing Vegard tensor, Vegard tensor not
of the form diag(b11, b11, b33), failed preparation, failed quadrature. Always
labelled approximate. Validated against the exact static isotropic integral,
the high-frequency limit, and an independent reconstruction from Appendix B8 and
C1 of the paper.

### 4.4 Prepared but not installed (Patch 4)

Bundle `~/.codex/.chatgpt-projects/g-p-6a8b.../outputs/p00-patch4` (and the ZIP
beside it) adds seven new files, edits none: `analytical.py` (explicit model
configurations), `electrostriction.py` (closed-form electrostriction plus
internal Maxwell stress under a point-charge field of explicit length l,
`u = 2 (A_ES + A_M) U_dc V_ac`), `piezoresponse.py` (bulk-d half-space centre
response, `u = d_eff V_ac`), `electrostatic.py` (sphere-apex capacitance
gradient with dielectric factor and explicit vacuum gap g, `F_1 = -C_g U_dc
V_ac`), two test files, one physics note. Verified at 692 passing tests in an
isolated copy. The physics is summarised in `P00_PHYSICS.md`.

### 4.5 What P00 deliberately does not do

- No combination of mechanisms, no compatibility check, no transfer function:
  the two-springs and contact-resonance model exists only in the spatial route.
- Only 1w; no DC or 2w outputs even where trivially available.
- Vegard refuses any DC bias; the Patch 4 quadratic models need DC to give 1w.
- Inputs still go through `ForwardInputs`, whose validation carries spatial and
  P0 requirements (for example a diffusivity is required by the default spatial
  backend). A P00-specific input route is noted as future work in every patch.
- No comparison with any measurement. All numbers are internal consistency.

## 5. Reusable pieces in the oracle (to re-derive, not necessarily to import)

| Need in the new repo | Oracle source | Comment |
|---|---|---|
| Contact radius and stiffness | `physics/contact.py` | Hertz, DMT, JKR, Tabor check, indentation modulus from the stiffness tensor |
| Screening scales | `physics/screening.py` (`debye`, `lattice_gas`) | P00 uses only `debye` explicitly |
| Surface-layer voltage divider | `physics/interface_capacitance.py` | static planar series capacitor |
| Capacitance gradient C' | `physics/tip_capacitance.py` | sphere, Hudlet cone plus lever, or calibrated value |
| Instrument transfer | `physics/detection.py` | `z = (k_c u + F)/(k_l + k_c)`, `H_CR = 1/(1 - r^2 + i r/Q)`, detector factor |
| Material card | `sample/material.py`, `sample/examples.py` | LiCoO2 and SrTiO3 example states with DOIs and confidence |
| Envelope and regime numbers | `core/applicability.py`, `core/regime.py` | the `REGIME_DESCRIPTIONS` table is reusable prose |
| Morozovska disk integral | `physics/p00/vegard.py` | already minimal and tested |

The oracle also ships the harmonic Vegard/electrostatic/thermal/electromechanical
machinery and the elastic solver; the meeting explicitly excludes those from the
MLP.

## 6. The patch approach that worked

The oracle was developed through the ChatGPT project "ESM_predictor" (local
mirror `~/.codex/.chatgpt-projects/g-p-6a8b01e67b2c8191b95baef576b5feec/`). Each
step is delivered as a self-contained bundle in `outputs/<step>/`:

- `README.md` with scope, what is added or edited, how to apply, and the limits;
- `VERIFICATION.md` with test counts before and after, environment versions,
  application checks and a five-part final review (concept, physics,
  integration, dependency boundaries, backwards compatibility);
- `manifest.json` and `baseline.json` with SHA-256 hashes of every existing file
  the patch depends on and every produced file;
- a raw Git patch plus full copies under `files/`;
- a guarded `apply_patchN.py` (check-only by default, refuses to overwrite,
  refuses changed context, no force option);
- a physics or decision note in `docs/` that goes into the repository.

Habits that produced the quality: patches that only add files; mechanisms
registered explicitly rather than discovered; dependency-boundary tests that run
in a fresh interpreter and block forbidden imports; independent analytical-limit
tests rather than tests that re-run the production algebra; explicit statements
of what remains unvalidated. Also in that folder: `esm_signal_origin_analysis_
skill.md`, a prompt skill for mechanism-hypothesis ranking (interpretation
side), and `ESM_signal_formation_report.docx`, the literature corpus and
mechanism taxonomy with 18 references.

## 7. Gap between the meeting's P0 target and the oracle's P00

1. **Instrument stage is missing in P00.** The target is `Y = R x H`; P00 stops
   at R. The two-springs plus contact-resonance model from `detection.py` is the
   obvious H for the MLP.
2. **Combination is missing in P00.** The target says "additive"; P00 refuses to
   sum by design because its mechanisms have incompatible boundary conditions.
   The new repo must decide the combination rule and state its assumption.
3. **Electrostatic term is a force, Vegard is a displacement.** They meet only
   after the instrument stage: `z = (k_c u_Vegard + F_es)/(k_l + k_c) x H_CR`.
4. **DC bias.** The MLP is a zero-DC AC sine, so the Vegard refusal of DC is not
   a blocker. The electrostatic 1w term then needs CPD or DC; at V_dc = V_cpd
   only its 2w term survives. Decide whether 2w is reported.
5. **Inputs are too heavy.** `ForwardInputs` requires spatial or P0 validity
   and 30-plus fields. The MLP needs a material card, an instrument card, a
   protocol (sine waveform) and an interface block.
6. **Uncertainty.** The card today stores 1-sigma plus a confidence label; the
   meeting asks for marginal distributions per parameter so that propagation and
   later inference are possible.
7. **Naming.** P0 and P00 mean different things in the meeting and in the
   oracle. Pick names once.
8. **No measured data anywhere.** Even the MLP demonstration should name the
   dataset, or synthetic data, it will be compared against.

## 8. Proposed shape of 2026_ESM_P0 (structure only)

A standalone package, not an import of `esm_oracle`; copy the few formulas and
keep the schemas lean. Suggested top level:

```
2026_ESM_P0/
  README.md            vocabulary (P0 / state+response / P1), scope, conventions
  docs/
    PHYSICS.md         the MLP formulas with signs and units (seed: P00_PHYSICS.md)
    APPLICABILITY.md   regime numbers, thresholds, what each mechanism refuses
    PROTOCOLS.md       action space; which model level each waveform needs
    DECISIONS.md       short architecture decision records
  esm_p0/
    cards/             material card, instrument card, interface, protocol
    prepare/           contact radius, screening scales, voltage divider, C'
    material/          mechanisms: vegard_morozovska, electrostatic_capacitor
                       (each: compute -> response + validity + regime numbers)
    instrument/        two springs, contact resonance, detector factor -> Y
    compose/           explicit additive rule with compatibility statement
    report/            prediction with breakdown, worst status, provenance
    uncertainty/       (optional in MLP) sampling of marginals through the chain
  tests/               analytical limits, sign conventions, JSON round trip,
                       refusal paths, import boundaries
  notebooks/           one end-to-end demo per material (frequency sweep)
```

Decisions to take before any code is written:

- vocabulary for the levels, and the name of the package;
- uncertainty representation on the card (normal / lognormal with parameters,
  or samples) and whether the MLP propagates it;
- electrostatic model for the MLP (sphere plus explicit gap, Hudlet cone plus
  lever, or a calibrated C');
- whether 2w and DC-biased 1w electrostatics are reported in the MLP;
- combination rule and its stated assumption (same readout point, same
  harmonic convention, instrument stage applied once);
- which materials demonstrate it (oracle fixtures: LiCoO2 with its placeholder
  permittivity and non-dilute ions; SrTiO3 with frozen vacancies and a
  non-isotropic Q) and which data, measured or synthetic, they are compared to;
- the hand-off format from Faith's literature agent to the material card.

## 9. Housekeeping

`2026_ESM_P0` is an empty directory and is not a Git repository. Its parent
`/Users/borisslautin` is itself a Git repository (remote
`kbarakati/camm_hackathon.git`, branch k4my4r), so any `git` command run inside
the new folder today acts on the home-directory repository. Run `git init` in
`2026_ESM_P0` before anything else.
