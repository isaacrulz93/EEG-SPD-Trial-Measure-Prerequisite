# Frozen protocol

## Representations

- B1: full-trial, train-reference AIRM tangent space, StandardScaler, LSQR LDA
  with automatic shrinkage.
- B2: full-trial AIRM MDM with label-dependent means fit on training trials.
- B3: window tangent shrinkage LDA; mean test-window decision score is primary,
  with majority vote reported separately.
- B4: trial mean of window log-covariance svec vectors, shrinkage LDA.
- B5: per-direction mean and population standard deviation using the exact P1/P2
  fixed bank, shrinkage LDA.
- P1: direct fixed-bank SPDSW distance to a deterministic, class- and
  subject-balanced pooled class measure. It is not a barycenter or prototype.
- P2: exact per-direction order statistics (L x W) with no interpolation and no
  reference-vector subtraction, shrinkage LDA.
- Ordered control: W2 ordered non-overlapping tangent features, shrinkage LDA;
  interpretation-only.

Every affine classifier is exactly `StandardScaler()` followed by
`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`.

## Protocols

- Gate 1A: stratified, trial-grouped five-fold CV within every subject/session;
  folds are averaged within session, then the two sessions within subject.
- Gate 1B: session 1 to 2 and session 2 to 1, averaged within subject.
- Gate 2 `no_ra`: fully inductive LOSO.
- Gate 2 `subject_ra`: LOSO after per-subject AIRM recentering. Target centering
  uses all of the target's unlabeled trials and is **TRANSDUCTIVE CALIBRATION**.

LOSO training trials are deterministically balanced across every source
subject x class cell. P1 uses this balancing in every protocol. Overlapping
windows never leave their source trial.

## Fixed banks and gates

The five Frobenius-uniform symmetric banks have `L=200` and seeds
`[0,1,2,3,4]`. They are generated once and reused across every fold and
inference call. Each bank is reported separately.

Only W1 gates. A P2 stage requires all three:

1. median across bank-wise mean subject deltas `P2-B5 >= 0.010`;
2. positive subject-median delta for at least 6/9 subjects;
3. positive mean delta for at least 4/5 banks.

Gate 1B additionally needs median bank-wise mean `P2-B1 >= -0.010`. Gate 2
subject-RA additionally needs `P2-B1 >= 0.000`. P1 is diagnosed analogously
against B2 but never terminates P2. Gate 0 is descriptive unless numerical
covariance/log validation fails.

The final positive designation conservatively requires P2 to pass all four
reported stages. A P2 failure yields `close`; P2 pass with P1 failure yields
`quantile-only`; both pass yields `SPDSW-readout`.

