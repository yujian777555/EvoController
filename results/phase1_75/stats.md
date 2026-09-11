# Phase 1.75 statistical report

- Control arm: `mlp2_closed_loop_normalized`; 20 evaluation seeds per arm x problem.
- Delta columns are `median(control - comparison)` with a percentile bootstrap 95% CI; positive favours the control (both metrics are maximized).
- `p` is the one-sided Wilcoxon signed-rank test (`greater`, `zero_method=zsplit`); `p_Holm` is the Holm-Bonferroni adjustment across the 5 problems within each (metric, comparison) family.

## final_hv — primary comparisons

| problem | vs static_full_global: delta_med | vs static_full_global: 95% CI | vs static_full_global: p | vs static_full_global: p_Holm | vs open_loop_global: delta_med | vs open_loop_global: 95% CI | vs open_loop_global: p | vs open_loop_global: p_Holm | vs generation_only_mlp: delta_med | vs generation_only_mlp: 95% CI | vs generation_only_mlp: p | vs generation_only_mlp: p_Holm | vs state_scrambled_mlp: delta_med | vs state_scrambled_mlp: 95% CI | vs state_scrambled_mlp: p | vs state_scrambled_mlp: p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zdt1 | -0.005449 | [-0.006797, -0.004449] | 1 | 1 | -0.002187 | [-0.003841, -0.001513] | 0.9999 | 1 | -0.005628 | [-0.007272, -0.004781] | 1 | 1 | -0.004036 | [-0.005164, -0.002662] | 1 | 1 |
| zdt2 | -0.004879 | [-0.00565, -0.003655] | 1 | 1 | 0.0009942 | [-0.001133, 0.005034] | 0.04865 | 0.1946 | -0.004393 | [-0.005597, -0.0035] | 1 | 1 | -0.001032 | [-0.002616, 0.001528] | 0.8256 | 1 |
| zdt3 | -0.005929 | [-0.008209, -0.004023] | 1 | 1 | -0.001584 | [-0.003581, -0.0003471] | 0.9972 | 1 | -0.006926 | [-0.009355, -0.005674] | 1 | 1 | -0.005369 | [-0.007381, -0.004197] | 1 | 1 |
| zdt4 | 0.3937 | [0.276, 0.6113] | 9.537e-07 | 4.768e-06 | 0.07671 | [0.0003797, 0.2273] | 0.1012 | 0.3037 | 0.04546 | [-0.1061, 0.1916] | 0.2608 | 1 | 0.1161 | [0.02441, 0.2445] | 0.04128 | 0.1651 |
| zdt6 | -0.001622 | [-0.003372, -0.0008985] | 1 | 1 | 0.003593 | [0.001937, 0.004234] | 0.0006046 | 0.003023 | -0.0007409 | [-0.001841, -0.0004447] | 1 | 1 | 0.006946 | [0.003991, 0.01126] | 3.147e-05 | 0.0001574 |

## auc_hv — primary comparisons

| problem | vs static_full_global: delta_med | vs static_full_global: 95% CI | vs static_full_global: p | vs static_full_global: p_Holm | vs open_loop_global: delta_med | vs open_loop_global: 95% CI | vs open_loop_global: p | vs open_loop_global: p_Holm | vs generation_only_mlp: delta_med | vs generation_only_mlp: 95% CI | vs generation_only_mlp: p | vs generation_only_mlp: p_Holm | vs state_scrambled_mlp: delta_med | vs state_scrambled_mlp: 95% CI | vs state_scrambled_mlp: p | vs state_scrambled_mlp: p_Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zdt1 | -0.02881 | [-0.03401, -0.02087] | 1 | 1 | 0.03734 | [0.02526, 0.04888] | 9.537e-07 | 4.768e-06 | -0.003336 | [-0.01367, 0.0002619] | 0.9621 | 1 | -0.001607 | [-0.009769, 0.005669] | 0.751 | 1 |
| zdt2 | -0.04009 | [-0.0451, -0.0344] | 1 | 1 | 0.07969 | [0.04794, 0.1412] | 9.537e-07 | 4.768e-06 | -0.03114 | [-0.03824, -0.02234] | 1 | 1 | 0.01317 | [0.001219, 0.0144] | 0.0009928 | 0.003971 |
| zdt3 | -0.03802 | [-0.05004, -0.02791] | 1 | 1 | 0.04337 | [0.02758, 0.05148] | 6.676e-06 | 2.003e-05 | -0.03307 | [-0.03778, -0.02467] | 1 | 1 | -0.002366 | [-0.01819, 0.01266] | 0.7392 | 1 |
| zdt4 | 0.07181 | [0.04157, 0.1665] | 9.537e-07 | 4.768e-06 | 0.03558 | [0.00482, 0.1238] | 0.00243 | 0.00243 | 0.01147 | [-0.03579, 0.07306] | 0.2262 | 1 | 0.04277 | [-0.0007223, 0.09489] | 0.004154 | 0.01246 |
| zdt6 | -0.0543 | [-0.06801, -0.04223] | 1 | 1 | 0.02844 | [0.01175, 0.04952] | 1.335e-05 | 2.67e-05 | -0.02633 | [-0.04557, -0.01358] | 1 | 1 | 0.04131 | [0.02719, 0.04865] | 8.392e-05 | 0.0004196 |

## Failure rates by arm x problem

| arm | zdt1 | zdt2 | zdt3 | zdt4 | zdt6 | overall |
|---|---|---|---|---|---|---|
| fixed_nsga2 | 0.15 | 0.25 | 0 | 0.5 | 0.15 | 0.21 |
| static_full_global | 0 | 0 | 0 | 1 | 0 | 0.2 |
| static_full_per_problem | 0 | 0 | 0 | 0.3 | 0 | 0.06 |
| open_loop_global | 0 | 0 | 0 | 0.2 | 0 | 0.04 |
| open_loop_per_problem | 0 | 0 | 0 | 0.8 | 0 | 0.16 |
| mlp2_closed_loop_absolute | 0 | 0 | 0 | 0.15 | 0 | 0.03 |
| mlp2_closed_loop_normalized | 0 | 0 | 0 | 0.15 | 0 | 0.03 |
| generation_only_mlp | 0 | 0 | 0 | 0.05 | 0 | 0.01 |
| state_scrambled_mlp | 0 | 0 | 0 | 0.3 | 0 | 0.06 |
