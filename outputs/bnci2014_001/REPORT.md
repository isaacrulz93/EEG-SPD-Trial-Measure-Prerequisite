1. Gate 1A status: FAIL
2. Gate 1B status: FAIL
3. Gate 2 no-RA status: FAIL
4. Gate 2 RA status: FAIL (TRANSDUCTIVE CALIBRATION)
5. final case: close

# Trial-measure prerequisite report

This is a no-neural-network prerequisite check. It contains no SPDHSW layer, TTA transport, source-memory alignment, or reference search. Gate 0 and P1 are descriptive/non-stopping; all cheap diagnostics were run after the covariance/log cache was valid.

## Data and preprocessing

- Raw epoch tensor shape: `(5184, 22, 1001)`
- Sampling frequency: `250.0 Hz`
- Actual sampled duration: `4.004000 s` (endpoint span `4.000000 s`)
- W1 exact windows/trial: `7`; total `36288`
- W2 exact windows/trial: `4`; total `20736`
- Covariance estimator: `oas_trace_eps1e-6` = OAS, trace division, `1e-6 I`, symmetrization.
- Filtering was performed once by the frozen MOABB `MotorImagery` paradigm on each full epoch before windowing.

## Gate 0 — descriptive only

| setting   |   within_trial_dispersion |   between_trial_dispersion |   ratio_within |   isotropic_fraction |   traceless_shape_fraction |   pca_effective_rank |   pc1_fraction |
|:----------|--------------------------:|---------------------------:|---------------:|---------------------:|---------------------------:|---------------------:|---------------:|
| W1        |                   3.83474 |                    4.90048 |       0.438997 |             0.10158  |                   0.89842  |              40.0622 |       0.126125 |
| W2        |                   3.91324 |                    4.82503 |       0.447828 |             0.102957 |                   0.897043 |              39.0218 |       0.128205 |

No dispersion, effective-rank, or PC1 threshold was used to stop the experiment.

## W1 primary classification summary

| protocol       | ra_mode    | method      |   balanced_accuracy |
|:---------------|:-----------|:------------|--------------------:|
| loso           | no_ra      | B1          |              0.3561 |
| loso           | no_ra      | B2          |              0.3806 |
| loso           | no_ra      | B3          |              0.3837 |
| loso           | no_ra      | B3_majority |              0.3929 |
| loso           | no_ra      | B4          |              0.3706 |
| loso           | no_ra      | B5          |              0.3948 |
| loso           | no_ra      | P1          |              0.3716 |
| loso           | no_ra      | P2          |              0.3787 |
| loso           | subject_ra | B1          |              0.5096 |
| loso           | subject_ra | B2          |              0.4985 |
| loso           | subject_ra | B3          |              0.5056 |
| loso           | subject_ra | B3_majority |              0.4855 |
| loso           | subject_ra | B4          |              0.4898 |
| loso           | subject_ra | B5          |              0.4969 |
| loso           | subject_ra | P1          |              0.4880 |
| loso           | subject_ra | P2          |              0.4574 |
| s1_to_s2       | no_ra      | B1          |              0.6690 |
| s1_to_s2       | no_ra      | B2          |              0.5918 |
| s1_to_s2       | no_ra      | B3          |              0.6574 |
| s1_to_s2       | no_ra      | B3_majority |              0.6385 |
| s1_to_s2       | no_ra      | B4          |              0.6593 |
| s1_to_s2       | no_ra      | B5          |              0.6263 |
| s1_to_s2       | no_ra      | P1          |              0.5694 |
| s1_to_s2       | no_ra      | P2          |              0.6145 |
| s2_to_s1       | no_ra      | B1          |              0.6030 |
| s2_to_s1       | no_ra      | B2          |              0.5872 |
| s2_to_s1       | no_ra      | B3          |              0.6215 |
| s2_to_s1       | no_ra      | B3_majority |              0.6026 |
| s2_to_s1       | no_ra      | B4          |              0.6130 |
| s2_to_s1       | no_ra      | B5          |              0.5948 |
| s2_to_s1       | no_ra      | P1          |              0.5640 |
| s2_to_s1       | no_ra      | P2          |              0.5845 |
| within_session | no_ra      | B1          |              0.7038 |
| within_session | no_ra      | B2          |              0.6381 |
| within_session | no_ra      | B3          |              0.7181 |
| within_session | no_ra      | B3_majority |              0.6892 |
| within_session | no_ra      | B4          |              0.6928 |
| within_session | no_ra      | B5          |              0.6579 |
| within_session | no_ra      | P1          |              0.5790 |
| within_session | no_ra      | P2          |              0.6645 |

Every projection-bank row is a separate fixed-bank result. Medians across seeds 0–4 are robustness summaries, not prediction ensembles.

### W1 fixed-bank results reported separately

| protocol       | ra_mode    | method   |   bank_seed |   balanced_accuracy |
|:---------------|:-----------|:---------|------------:|--------------------:|
| loso           | no_ra      | B5       |      0.0000 |              0.3929 |
| loso           | no_ra      | B5       |      1.0000 |              0.4059 |
| loso           | no_ra      | B5       |      2.0000 |              0.3912 |
| loso           | no_ra      | B5       |      3.0000 |              0.3895 |
| loso           | no_ra      | B5       |      4.0000 |              0.3945 |
| loso           | no_ra      | P1       |      0.0000 |              0.3684 |
| loso           | no_ra      | P1       |      1.0000 |              0.3760 |
| loso           | no_ra      | P1       |      2.0000 |              0.3688 |
| loso           | no_ra      | P1       |      3.0000 |              0.3858 |
| loso           | no_ra      | P1       |      4.0000 |              0.3592 |
| loso           | no_ra      | P2       |      0.0000 |              0.3715 |
| loso           | no_ra      | P2       |      1.0000 |              0.3808 |
| loso           | no_ra      | P2       |      2.0000 |              0.3902 |
| loso           | no_ra      | P2       |      3.0000 |              0.3682 |
| loso           | no_ra      | P2       |      4.0000 |              0.3825 |
| loso           | subject_ra | B5       |      0.0000 |              0.4865 |
| loso           | subject_ra | B5       |      1.0000 |              0.5010 |
| loso           | subject_ra | B5       |      2.0000 |              0.5027 |
| loso           | subject_ra | B5       |      3.0000 |              0.4873 |
| loso           | subject_ra | B5       |      4.0000 |              0.5071 |
| loso           | subject_ra | P1       |      0.0000 |              0.4973 |
| loso           | subject_ra | P1       |      1.0000 |              0.4821 |
| loso           | subject_ra | P1       |      2.0000 |              0.4794 |
| loso           | subject_ra | P1       |      3.0000 |              0.4983 |
| loso           | subject_ra | P1       |      4.0000 |              0.4832 |
| loso           | subject_ra | P2       |      0.0000 |              0.4473 |
| loso           | subject_ra | P2       |      1.0000 |              0.4695 |
| loso           | subject_ra | P2       |      2.0000 |              0.4618 |
| loso           | subject_ra | P2       |      3.0000 |              0.4452 |
| loso           | subject_ra | P2       |      4.0000 |              0.4632 |
| s1_to_s2       | no_ra      | B5       |      0.0000 |              0.6304 |
| s1_to_s2       | no_ra      | B5       |      1.0000 |              0.6296 |
| s1_to_s2       | no_ra      | B5       |      2.0000 |              0.6223 |
| s1_to_s2       | no_ra      | B5       |      3.0000 |              0.6130 |
| s1_to_s2       | no_ra      | B5       |      4.0000 |              0.6362 |
| s1_to_s2       | no_ra      | P1       |      0.0000 |              0.5725 |
| s1_to_s2       | no_ra      | P1       |      1.0000 |              0.5856 |
| s1_to_s2       | no_ra      | P1       |      2.0000 |              0.5590 |
| s1_to_s2       | no_ra      | P1       |      3.0000 |              0.5652 |
| s1_to_s2       | no_ra      | P1       |      4.0000 |              0.5648 |
| s1_to_s2       | no_ra      | P2       |      0.0000 |              0.6192 |
| s1_to_s2       | no_ra      | P2       |      1.0000 |              0.6011 |
| s1_to_s2       | no_ra      | P2       |      2.0000 |              0.6258 |
| s1_to_s2       | no_ra      | P2       |      3.0000 |              0.6034 |
| s1_to_s2       | no_ra      | P2       |      4.0000 |              0.6231 |
| s2_to_s1       | no_ra      | B5       |      0.0000 |              0.5910 |
| s2_to_s1       | no_ra      | B5       |      1.0000 |              0.5907 |
| s2_to_s1       | no_ra      | B5       |      2.0000 |              0.5930 |
| s2_to_s1       | no_ra      | B5       |      3.0000 |              0.5907 |
| s2_to_s1       | no_ra      | B5       |      4.0000 |              0.6088 |
| s2_to_s1       | no_ra      | P1       |      0.0000 |              0.5706 |
| s2_to_s1       | no_ra      | P1       |      1.0000 |              0.5567 |
| s2_to_s1       | no_ra      | P1       |      2.0000 |              0.5656 |
| s2_to_s1       | no_ra      | P1       |      3.0000 |              0.5664 |
| s2_to_s1       | no_ra      | P1       |      4.0000 |              0.5610 |
| s2_to_s1       | no_ra      | P2       |      0.0000 |              0.5710 |
| s2_to_s1       | no_ra      | P2       |      1.0000 |              0.5934 |
| s2_to_s1       | no_ra      | P2       |      2.0000 |              0.5926 |
| s2_to_s1       | no_ra      | P2       |      3.0000 |              0.5729 |
| s2_to_s1       | no_ra      | P2       |      4.0000 |              0.5926 |
| within_session | no_ra      | B5       |      0.0000 |              0.6621 |
| within_session | no_ra      | B5       |      1.0000 |              0.6561 |
| within_session | no_ra      | B5       |      2.0000 |              0.6558 |
| within_session | no_ra      | B5       |      3.0000 |              0.6589 |
| within_session | no_ra      | B5       |      4.0000 |              0.6566 |
| within_session | no_ra      | P1       |      0.0000 |              0.5865 |
| within_session | no_ra      | P1       |      1.0000 |              0.5835 |
| within_session | no_ra      | P1       |      2.0000 |              0.5743 |
| within_session | no_ra      | P1       |      3.0000 |              0.5806 |
| within_session | no_ra      | P1       |      4.0000 |              0.5703 |
| within_session | no_ra      | P2       |      0.0000 |              0.6619 |
| within_session | no_ra      | P2       |      1.0000 |              0.6716 |
| within_session | no_ra      | P2       |      2.0000 |              0.6635 |
| within_session | no_ra      | P2       |      3.0000 |              0.6648 |
| within_session | no_ra      | P2       |      4.0000 |              0.6609 |

## W2 sensitivity summary (non-gating)

| protocol       | ra_mode    | method          |   balanced_accuracy |
|:---------------|:-----------|:----------------|--------------------:|
| loso           | no_ra      | B1              |              0.3561 |
| loso           | no_ra      | B2              |              0.3806 |
| loso           | no_ra      | B3              |              0.3779 |
| loso           | no_ra      | B3_majority     |              0.3862 |
| loso           | no_ra      | B4              |              0.3673 |
| loso           | no_ra      | B5              |              0.3845 |
| loso           | no_ra      | P1              |              0.3636 |
| loso           | no_ra      | P2              |              0.3741 |
| loso           | no_ra      | ordered_control |              0.3945 |
| loso           | subject_ra | B1              |              0.5096 |
| loso           | subject_ra | B2              |              0.4985 |
| loso           | subject_ra | B3              |              0.4998 |
| loso           | subject_ra | B3_majority     |              0.4599 |
| loso           | subject_ra | B4              |              0.4770 |
| loso           | subject_ra | B5              |              0.4855 |
| loso           | subject_ra | P1              |              0.4751 |
| loso           | subject_ra | P2              |              0.4629 |
| loso           | subject_ra | ordered_control |              0.5017 |
| s1_to_s2       | no_ra      | B1              |              0.6690 |
| s1_to_s2       | no_ra      | B2              |              0.5918 |
| s1_to_s2       | no_ra      | B3              |              0.6458 |
| s1_to_s2       | no_ra      | B3_majority     |              0.6022 |
| s1_to_s2       | no_ra      | B4              |              0.6532 |
| s1_to_s2       | no_ra      | B5              |              0.6070 |
| s1_to_s2       | no_ra      | P1              |              0.5529 |
| s1_to_s2       | no_ra      | P2              |              0.6059 |
| s1_to_s2       | no_ra      | ordered_control |              0.6524 |
| s2_to_s1       | no_ra      | B1              |              0.6030 |
| s2_to_s1       | no_ra      | B2              |              0.5872 |
| s2_to_s1       | no_ra      | B3              |              0.6169 |
| s2_to_s1       | no_ra      | B3_majority     |              0.5772 |
| s2_to_s1       | no_ra      | B4              |              0.5992 |
| s2_to_s1       | no_ra      | B5              |              0.5837 |
| s2_to_s1       | no_ra      | P1              |              0.5487 |
| s2_to_s1       | no_ra      | P2              |              0.5789 |
| s2_to_s1       | no_ra      | ordered_control |              0.6296 |
| within_session | no_ra      | B1              |              0.7038 |
| within_session | no_ra      | B2              |              0.6381 |
| within_session | no_ra      | B3              |              0.6982 |
| within_session | no_ra      | B3_majority     |              0.6540 |
| within_session | no_ra      | B4              |              0.6757 |
| within_session | no_ra      | B5              |              0.6403 |
| within_session | no_ra      | P1              |              0.5620 |
| within_session | no_ra      | P2              |              0.6458 |
| within_session | no_ra      | ordered_control |              0.6947 |

W2 never rescues or overturns a W1 decision.

## Gate details

```json
{
  "approximately_equal_tolerance": 0.01,
  "final_case": "close",
  "gate_1a": {
    "name": "Gate 1A",
    "p1_diagnostic_vs_b2": {
      "bank_mean_deltas": {
        "0": -0.0516837701270751,
        "1": -0.05463147740011775,
        "2": -0.06381833591102169,
        "3": -0.05752812762906934,
        "4": -0.06788059035382307
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.05752812762906934,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 0,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.03620973034962971,
        "2": -0.0677711718965589,
        "3": -0.08099552107679042,
        "4": -0.06326247700957449,
        "5": -0.03979751092560685,
        "6": -0.06043110183590067,
        "7": -0.08698579961060615,
        "8": -0.03813245470559723,
        "9": -0.051805713636209005
      },
      "subject_wins": 0
    },
    "p1_failure_stops_p2": false,
    "p2_pass": false,
    "p2_vs_b1": null,
    "p2_vs_b1_requirement_pass": true,
    "p2_vs_b1_threshold": null,
    "p2_vs_b5": {
      "bank_mean_deltas": {
        "0": -0.00022706009161118892,
        "1": 0.015472293325105524,
        "2": 0.007652713917871275,
        "3": 0.005820313420287598,
        "4": 0.0043096164193945664
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": 0.005820313420287598,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 4,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.0031006339936293648,
        "2": 0.025200747395019818,
        "3": 0.0007542381294316503,
        "4": -0.005969201829302384,
        "5": 0.004515381956364917,
        "6": 0.006639998503388589,
        "7": 0.019326168220324735,
        "8": -0.004245675742386212,
        "9": -0.0016867379696325902
      },
      "subject_wins": 5
    },
    "primary_window_setting": "W1",
    "protocol": "within_session",
    "ra_mode": "no_ra",
    "status": "FAIL"
  },
  "gate_1b": {
    "name": "Gate 1B",
    "p1_diagnostic_vs_b2": {
      "bank_mean_deltas": {
        "0": -0.017939814814814863,
        "1": -0.01832561728395062,
        "2": -0.027199074074074108,
        "3": -0.023726851851851884,
        "4": -0.026620370370370388
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.023726851851851884,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 0,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.02430555555555558,
        "2": -0.046874999999999944,
        "3": -0.01909722222222232,
        "4": -0.02604166666666674,
        "5": -0.006944444444444475,
        "6": 0.01215277777777779,
        "7": -0.02604166666666663,
        "8": -0.02430555555555558,
        "9": -0.01041666666666674
      },
      "subject_wins": 1
    },
    "p1_failure_stops_p2": false,
    "p2_pass": false,
    "p2_vs_b1": {
      "bank_mean_deltas": {
        "0": -0.04089506172839505,
        "1": -0.03877314814814816,
        "2": -0.026813271604938255,
        "3": -0.04783950617283949,
        "4": -0.028163580246913584
      },
      "median_across_banks_mean_subject_delta": -0.03877314814814816
    },
    "p2_vs_b1_requirement_pass": false,
    "p2_vs_b1_threshold": -0.01,
    "p2_vs_b5": {
      "bank_mean_deltas": {
        "0": -0.015625000000000014,
        "1": -0.01292438271604938,
        "2": 0.0015432098765431983,
        "3": -0.013695987654320983,
        "4": -0.014660493827160516
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.013695987654320983,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 1,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.01736111111111116,
        "2": 0.01388888888888884,
        "3": 0.00520833333333337,
        "4": -0.01041666666666663,
        "5": -0.01909722222222221,
        "6": 0.001736111111111105,
        "7": -0.00694444444444442,
        "8": -0.03125,
        "9": -0.02777777777777779
      },
      "subject_wins": 3
    },
    "primary_window_setting": "W1",
    "protocol": "cross_session",
    "ra_mode": "no_ra",
    "status": "FAIL"
  },
  "gate_2_no_ra": {
    "name": "Gate 2 no_ra",
    "p1_diagnostic_vs_b2": {
      "bank_mean_deltas": {
        "0": -0.01215277777777777,
        "1": -0.004629629629629626,
        "2": -0.01176697530864197,
        "3": 0.005208333333333355,
        "4": -0.021412037037037045
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.01176697530864197,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 1,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.02256944444444442,
        "2": 0.013888888888888895,
        "3": -0.07118055555555552,
        "4": -0.02777777777777779,
        "5": 0.0,
        "6": 0.005208333333333398,
        "7": -0.13020833333333337,
        "8": 0.09548611111111116,
        "9": 0.017361111111111105
      },
      "subject_wins": 4
    },
    "p1_failure_stops_p2": false,
    "p2_pass": false,
    "p2_vs_b1": null,
    "p2_vs_b1_requirement_pass": true,
    "p2_vs_b1_threshold": null,
    "p2_vs_b5": {
      "bank_mean_deltas": {
        "0": -0.021412037037037004,
        "1": -0.025077160493827175,
        "2": -0.0009645061728395089,
        "3": -0.02121913580246916,
        "4": -0.011959876543209878
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.02121913580246916,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 0,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.046875000000000056,
        "2": 0.001736111111111105,
        "3": -0.020833333333333315,
        "4": -0.00694444444444442,
        "5": 0.0034722222222222654,
        "6": -0.00520833333333337,
        "7": 0.00347222222222221,
        "8": -0.05555555555555558,
        "9": -0.022569444444444475
      },
      "subject_wins": 3
    },
    "primary_window_setting": "W1",
    "protocol": "loso",
    "ra_mode": "no_ra",
    "status": "FAIL"
  },
  "gate_2_subject_ra": {
    "name": "Gate 2 subject_ra",
    "p1_diagnostic_vs_b2": {
      "bank_mean_deltas": {
        "0": -0.0011574074074073787,
        "1": -0.0163966049382716,
        "2": -0.019097222222222234,
        "3": -0.00019290123456790055,
        "4": -0.015239197530864168
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.015239197530864168,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 0,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.00868055555555547,
        "2": -0.01041666666666663,
        "3": -0.00347222222222221,
        "4": 0.0,
        "5": -0.029513888888888784,
        "6": 0.03298611111111116,
        "7": -0.05729166666666663,
        "8": 0.005208333333333259,
        "9": -0.03298611111111105
      },
      "subject_wins": 2
    },
    "p1_failure_stops_p2": false,
    "p2_pass": false,
    "p2_vs_b1": {
      "bank_mean_deltas": {
        "0": -0.06230709876543212,
        "1": -0.04012345679012347,
        "2": -0.04783950617283951,
        "3": -0.06442901234567903,
        "4": -0.04648919753086419
      },
      "median_across_banks_mean_subject_delta": -0.04783950617283951
    },
    "p2_vs_b1_requirement_pass": false,
    "p2_vs_b1_threshold": 0.0,
    "p2_vs_b5": {
      "bank_mean_deltas": {
        "0": -0.03915895061728398,
        "1": -0.031442901234567944,
        "2": -0.04089506172839509,
        "3": -0.04205246913580246,
        "4": -0.04398148148148149
      },
      "complete_nine_subject_five_bank_suite": true,
      "delta_threshold": 0.01,
      "median_across_banks_mean_subject_delta": -0.04089506172839509,
      "n_banks": 5,
      "n_subjects": 9,
      "pass": false,
      "positive_mean_delta_banks": 0,
      "required_positive_banks": 4,
      "required_subject_wins": 6,
      "status": "FAIL",
      "subject_median_deltas": {
        "1": -0.06076388888888906,
        "2": -0.01215277777777779,
        "3": -0.0746527777777779,
        "4": -0.02604166666666663,
        "5": -0.019097222222222265,
        "6": -0.02083333333333326,
        "7": -0.029513888888888895,
        "8": -0.07465277777777779,
        "9": -0.05555555555555558
      },
      "subject_wins": 0
    },
    "primary_window_setting": "W1",
    "protocol": "loso",
    "ra_mode": "subject_ra",
    "status": "FAIL"
  },
  "overall_p1_diagnostic_pass_requires_all_four_stages": false,
  "overall_p2_pass_requires_all_four_stages": false,
  "w2_can_change_gate": false
}
```

## Interpretation controls

- P2 is approximately equal to B5 (absolute median-bank delta <= 0.010) for Gate 1A; at those stages moment pooling explains the result and higher quantiles are unnecessary.
- The ordered non-overlapping-window control exceeds P2 by +0.0390 mean BA in W2; temporal order is more useful than an exchangeable trial measure.
- Final decision: P1 diagnostic failed and P2 failed; therefore the registered action is `close` (close trial-measure SPDSW/SQE when `close`).

## Timing

- Covariance/log-cache time: `41.318436 s`
- Subject-RA covariance/log time: `17.554738 s`
- Projection-feature precomputation time: `10.273244 s`
- Classifier/pooled-class-measure fitting time: `2630.468082 s`
- Total measured inference time: `1670.479040 s`
- Amortized measured inference time: `0.001964863 s/trial`

Projection precomputation is reported separately from fitting and inference. Per-row fitting and amortized per-trial inference measurements are in `classification_results.csv`.

## Provenance and claim discipline

- Frozen MOABB factory source: `isaacrulz93/T3DA` at `7d0b8b1757c8284a6e9a641487d3cf6106073dc5`; file blob `010e445af8fcdf65f8719e465ff2ed2e3672e3ea`.
- `subject_ra` uses each target subject's own unlabeled trials and is explicitly transductive calibration, not fully inductive.
- P1 compares to a pooled class measure. It is not called a Wasserstein barycenter or prototype.
- No downstream neural-layer claim follows from this prerequisite experiment.
