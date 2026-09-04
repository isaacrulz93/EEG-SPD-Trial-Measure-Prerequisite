import pandas as pd

from src.gates import evaluate_stage


def _rows(delta=0.02, wins=9, positive_banks=5):
    rows = []
    for subject in range(1, 10):
        for seed in range(5):
            signed = delta if subject <= wins and seed < positive_banks else -abs(delta)
            common = dict(protocol="within_session", ra_mode="no_ra", window_setting="W1", subject=subject, bank_seed=seed)
            rows.append({**common, "method": "B5", "balanced_accuracy": 0.5})
            rows.append({**common, "method": "P2", "balanced_accuracy": 0.5 + signed})
            rows.append({**common, "method": "B2_bank", "balanced_accuracy": 0.5})
            rows.append({**common, "method": "P1", "balanced_accuracy": 0.5 + signed})
        rows.append({"protocol": "within_session", "ra_mode": "no_ra", "window_setting": "W1", "subject": subject, "bank_seed": None, "method": "B1", "balanced_accuracy": 0.5})
    return pd.DataFrame(rows)


def test_p2_gate_requires_all_three_conditions():
    assert evaluate_stage(_rows(), name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]
    assert not evaluate_stage(_rows(delta=0.009), name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]
    assert not evaluate_stage(_rows(wins=5), name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]
    assert not evaluate_stage(_rows(positive_banks=3), name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]


def test_w2_never_enters_primary_gate():
    frame = _rows()
    altered = frame.copy()
    altered["window_setting"] = "W2"
    altered.loc[altered.method == "P2", "balanced_accuracy"] = 1.0
    combined = pd.concat([frame, altered], ignore_index=True)
    assert evaluate_stage(frame, name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]
    assert evaluate_stage(combined, name="Gate 1A", protocol="within_session", ra_mode="no_ra")["p2_pass"]

