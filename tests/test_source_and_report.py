import inspect
from pathlib import Path

from experiments.slma_data import _import_moabb_dataset
from src.report import write_initial_report


def test_frozen_factory_has_exact_motor_imagery_construction():
    source = inspect.getsource(_import_moabb_dataset)
    assert 'MotorImagery(n_classes=cfg["n_classes"], fmin=cfg["fmin"], fmax=cfg["fmax"])' in source


def test_report_starts_with_mandatory_five_lines(tmp_path):
    path = tmp_path / "REPORT.md"
    write_initial_report(path)
    lines = path.read_text().splitlines()
    assert lines[:5] == [
        "1. Gate 1A status: NOT RUN",
        "2. Gate 1B status: NOT RUN",
        "3. Gate 2 no-RA status: NOT RUN",
        "4. Gate 2 RA status: NOT RUN",
        "5. final case: NOT RUN",
    ]


def test_no_neural_framework_is_imported():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(path.read_text() for path in [*root.glob("src/*.py"), root / "run_experiment.py"])
    assert "import torch" not in source
    assert "tensorflow" not in source

