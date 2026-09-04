# EEG SPD trial-measure prerequisite

This repository performs the complete cheap, no-neural-network prerequisite
check for trial-level exchangeable window measures on BNCI2014-001.  It does
not implement an SPDHSW layer, TTA transport, source-memory alignment, or a
reference search.

## Frozen data path

`experiments/slma_data.py` preserves the MOABB construction from private
repository `isaacrulz93/T3DA`, commit
`7d0b8b1757c8284a6e9a641487d3cf6106073dc5`, blob
`010e445af8fcdf65f8719e465ff2ed2e3672e3ea`.  The runner imports its
`_import_moabb_dataset` factory and makes the same single
`MotorImagery.get_data(dataset=dataset, subjects=all_subjects)` call. Filtering
therefore happens on each full epoch before either window definition is
applied; no second paradigm exists.

The primary covariance is OAS, divided by trace, regularized with `1e-6 I`,
then symmetrized. W1 is 1.0 s / 0.5 s hop and is the only gating window. W2 is
1.0 s / 1.0 s hop and is sensitivity-only.

## Run

The tested environment already available on this workstation is:

```bash
/home/pikachu/miniconda3/envs/spdhsw/bin/python -m pytest -q
/home/pikachu/miniconda3/envs/spdhsw/bin/python run_experiment.py
```

The first command must pass before a scientific run. Raw epochs are retained
in memory only until full-trial and W1/W2 covariance/log caches have been
written. The run is resumable at the protocol/fold/window level under
`outputs/bnci2014_001/runs/`.

## Outputs

The completed runner writes:

- `REPORT.md` and `outputs/bnci2014_001/REPORT.md`
- `classification_results.csv` with every mandatory classification column
- `gate0_diagnostics.csv` and `GATES.json`
- `timing_summary.csv`, with cache, projection, fitting, and inference timing
- `bank_audit.json`, cache validation, and provenance
- one CSV per protocol/fold/window under `runs/`, including null results

Each bank seed is reported separately. A across-bank median is a robustness
summary and never an ensemble. `subject_ra` is labeled **TRANSDUCTIVE
CALIBRATION** because it uses each target subject's own unlabeled trial set.

See [PROTOCOL.md](PROTOCOL.md) for the exact classifiers and gate equations.

