"""esm_p00: minimal forward P00 model for electrochemical strain microscopy.

Four independent analytical mechanisms (Vegard, piezoresponse, electrostriction,
electrostatic force) evaluated at the disk centre under one shared set of
conventions, and one linear detector that turns displacement and force into
the measured signal. Mechanisms never import each other; shared scales come
only from `scales.py`; combination happens only in the detector, assembled in
a notebook. See docs/CONVENTIONS.md and docs/PHYSICS.md.
"""
