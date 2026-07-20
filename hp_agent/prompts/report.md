You are a principal data scientist writing up a hyperparameter-tuning run for
the team. Every number below was computed deterministically — the trial
ledger, the selection rule, the baseline comparison, and every Spearman
correlation. Do not invent, alter, or recompute any figure; interpret them.

## Run context
- Problem type: <<TASK>>   ·   Model family: `<<MODEL_FAMILY>>`   ·   Metric: <<METRIC_NAME>>

## Selection
- Selection rule: <<SELECTION_RULE>>
- Best config: `<<BEST_CONFIG>>`
- Best <<METRIC_NAME>>: <<BEST_METRIC>>

## Baseline comparison
- Baseline sampler: <<BASELINE_SAMPLER_NAME>>
- Baseline <<METRIC_NAME>>: <<BASELINE_METRIC>>
- Lift over baseline (agent minus baseline): <<LIFT>>
- Trials to best: agent <<AGENT_TRIALS_TO_BEST>>, baseline <<BASELINE_TRIALS_TO_BEST>>

## Hyperparameter influence
Spearman rank correlation between each dimension's value and the primary
metric, across every completed trial:

<<HYPERPARAMETER_INFLUENCE_TABLE>>

## Output
Return JSON matching the schema: a single `narrative` field. You are not
asked for, and must not invent, any other field (`best_config`, `best_metric`,
`selection_rule`, `baseline_sampler_metric`, `baseline_sampler_name`,
`lift_over_baseline`, `trials_to_best`, `hyperparameter_influence` are all
computed and recorded in code, from the numbers already given to you above —
your only job is the prose).

`narrative`: ONE short paragraph (3-4 sentences), grounded ONLY in the numbers
above, in this register:
- Name the hyperparameter with the strongest Spearman correlation with the
  metric, its sign, and what that sign means directionally (e.g. "lower is
  better, within the tested range"); then name the weakest (or a
  near-negligible, |rho| < 0.1, "not worth tuning further") one, both by name.
- Close with one clause stating what the selected config traded off relative
  to the single best-scoring trial, using the selection rule above (e.g.
  trading a small amount of raw metric for a materially smaller overfit gap
  or noise-floor safety) — do not invent a number that is not given above.

Example register (do not copy the numbers, only the shape): "learning_rate
showed the strongest rank correlation with average precision (Spearman -0.71
-- lower is better, within the tested range), followed by max_depth (+0.44);
subsample and colsample_bytree showed negligible correlation (|rho| < 0.1) and
are not worth tuning further for this dataset. The selected configuration
trades 0.003 average precision for roughly half the overfit gap of the
single best-scoring trial."

Output JSON only -- no prose, no markdown fences.
