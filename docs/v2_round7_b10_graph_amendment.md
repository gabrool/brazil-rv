# Round 7 B10 pretraining graph-count amendment

At 2026-09-13 06:21 UTC, all 21 remaining original Stage-P fits had completed. Recursive comparison against the original launch contract found exactly one discrepancy in each B10 seed (11, 29, 47): two compiled training graphs instead of one. All 18 other fits matched their launch contracts. Selection graph counts were one throughout.

A separate diagnostic using the frozen implementation and accepted data, with TORCH_LOGS=recompiles, identified this compiler guard:

`2048*batch['active_mask'].size()[1]*batch['common_state_features'].size()[0] >= 5242880`

This is a compiler tensor-size specialization threshold. The diagnostic reproduced the second graph and continued across eight epochs before being stopped. Although --epochs 1 was requested, Stage P correctly forced its registered 60-epoch budget; the diagnostic was explicitly stopped and is not an additional research trajectory. Its output remains separate under engineering_b10_recompile and b10_recompile.log.

Under the user's explicit authority to resolve implementation choices, accept exactly the observed two training graphs and one selection graph for these three completed B10 P trajectories. Preserve the original failed launcher, original plan, manifests, checkpoints and graph counts. Validate an amended plan into a separate launcher, reusing completed fits without retraining. No model, optimizer, input, history, eligible-name set, score or checkpoint is changed. This is a disclosed relaxation of the engineering count criterion, not a claim that the original one-graph criterion passed. No financial performance was used to choose this amendment. Compiler logs diagnose the specialization trigger; they are not an independent numerical equivalence test.

This amendment applies only to these three Stage-P trajectories. Future graph-count failures require inspection; it does not silently relax the screen or other acceptance contracts. Both frozen compute checkouts remain untouched. The final combined review must disclose the amendment and measured compile overhead. Source evidence lives in the original frozen run root v2_round7_20260913T012300Z.
