# AI Runtime run 3 (warm start from the published edit), 3 Nov 2026

Same setup as run 2 but the GEPA seed candidate was the published Experiment-8 edit and the reflection model got
6 passes per round. Round 1 (16 turns, 7 code runs, 5 passes) found that each layer's published delta is exactly
rank-1, saved both deltas as artifacts, and swept the scale on 12 fresh practice problems, paired against the original:

| Delta scale | Accuracy | Looping | Cut at limit |
|---|---|---|---|
| 1.0x (published) | 0.25 | 0.50 | 0.83 |
| 1.6x | 0.25 | 0.25 | 0.83 |
| 2.4x | 0.33 | 0.00 | 0.42 |
| 3.2x | 0.00 | 0.08 | 0.92 |

It submitted 2.4x. GEPA accepted it on the minibatch and crashed during the full validation pass on a metric bug
(an answer of `1e999` overflowed `int(float(...))`), so the candidate was lost and the run was cancelled. Both the
metric and a parser gap (tool-call XML instead of fenced blocks, visible in turns 7, 8 and 13) are fixed in the
next commits; run 4 repeats this configuration and also scores the dose-response directly on the holdout.
