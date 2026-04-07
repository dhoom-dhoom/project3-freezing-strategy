# Project 3 Deliverable: Transfer Learning Layer Freezing Strategy

## Research Question

Which freezing strategy should be recommended when fine-tuning an ImageNet-pretrained `ResNet-18` on `IndiWASTE` under a fixed `900` second training budget?

## Experimental Setup

- Dataset: `IndiWASTE`
- Model: `ResNet-18` pretrained on ImageNet
- Runtime: Kaggle `Tesla P100`
- Budget per run: `900` seconds
- Primary metric: `val_error = 1 - val_macro_f1`
- Secondary metrics:
  - `val_macro_f1`
  - `val_accuracy`

All runs used the same data pipeline, transforms, evaluation logic, and hardware budget. Only the freezing strategy changed across experiments.

## Freezing Strategies Compared

- `all_but_head`
- `freeze_early`
- `freeze_late`
- `freeze_none`

## Results Table

| Strategy | Commit | Val Error | Val Macro F1 | Val Accuracy | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `all_but_head` | `ee98861` | `0.152654` | `0.847346` | `0.846325` | `keep` |
| `freeze_early` | `3979af4` | `0.157133` | `0.842867` | `0.844098` | `discard` |
| `freeze_late` | `a693722` | `0.166171` | `0.833829` | `0.832962` | `discard` |
| `freeze_none` | `1ee91b8` | `0.185411` | `0.814589` | `0.812918` | `discard` |

## Ranking

1. `all_but_head`
2. `freeze_early`
3. `freeze_late`
4. `freeze_none`

## Main Finding

The best strategy was `all_but_head`, which froze the pretrained backbone and trained only the classification head. It achieved the lowest `val_error` and the highest `val_macro_f1` among all four strategies.

## Interpretation

These results suggest that, for `IndiWASTE` and a short `900` second fine-tuning budget, aggressively updating more of the backbone does not help. In fact, performance consistently worsened as more pretrained layers were allowed to change.

Two practical reasons likely explain this:

- The dataset is relatively small, so larger trainable parameter counts make overfitting easier.
- The time budget is short, so heavily trainable models may not stabilize as effectively as a lightweight head-only adaptation.

This pattern is visible in the results:

- `freeze_early` was slightly worse than the baseline.
- `freeze_late` was meaningfully worse.
- `freeze_none`, which trained the entire network, was the worst configuration overall.

## Recommendation

For this project, the recommended transfer learning strategy is:

`Freeze the full pretrained backbone and train only the final classification head.`

This is the best choice because it:

- produced the strongest validation performance
- used the smallest trainable parameter set
- matched the short-budget setting well
- was the simplest and most stable strategy tested

## Deliverable Summary

The freezing-strategy comparison table and the recommendation are now complete for Project 3. Under a controlled `900` second budget on `IndiWASTE`, `all_but_head` should be recommended as the final strategy.
