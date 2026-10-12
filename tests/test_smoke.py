"""The package and every placeholder module import in the project venv."""
import importlib

MODULES = [
    "esm_p00", "esm_p00.constants", "esm_p00.cards", "esm_p00.scales",
    "esm_p00.report", "esm_p00.state", "esm_p00.detector",
    "esm_p00.mechanisms.vegard", "esm_p00.mechanisms.piezoresponse",
    "esm_p00.mechanisms.electrostriction", "esm_p00.mechanisms.electrostatic",
    "esm_p00.protocols", "esm_p00.examples", "esm_p00.particles", "material_state", "material_state.occupancy",
]


def test_imports():
    for name in MODULES:
        importlib.import_module(name)
