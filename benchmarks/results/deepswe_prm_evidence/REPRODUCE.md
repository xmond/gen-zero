Reproduction uses local files only, CPU execution, and four BLAS/PyTorch threads.
Run from `/ebs/pj/gen-zero-worktree/rm-r1-prm` with package versions recorded in
`environment.json`. No agents or reviewers were launched.

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python -u benchmarks/eval_deepswe_enhanced_prm.py prepare --partition train --output benchmarks/artifacts/deepswe_prm/train_features.pt
python -u benchmarks/eval_deepswe_enhanced_prm.py train --features benchmarks/artifacts/deepswe_prm/train_features.pt --output benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt
python -u benchmarks/eval_deepswe_enhanced_prm.py prepare --partition heldout --output benchmarks/artifacts/deepswe_prm/heldout_features.pt
python -u benchmarks/eval_deepswe_enhanced_prm.py evaluate --features benchmarks/artifacts/deepswe_prm/heldout_features.pt --checkpoint benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt --output benchmarks/results/deepswe_prm_enhanced_results.json
python -m unittest discover -s benchmarks/tests -p test_deepswe_enhanced_prm.py -v
python benchmarks/tests/verify_deepswe_prm_report.py --report benchmarks/results/deepswe_prm_enhanced_results.json --reference-repo /tmp/clmrepro/repo
```

The evaluator deliberately rejects an existing output. For an independent
reproduction, use a fresh output path; do not overwrite the recorded experiment
or use the heldout outcome to change training choices. There is no inference
fallback to the baseline. The baseline is computed separately from its real
checkpoint and uses the mean of its final 12 step cosine scores.

Each executed command has a JSON record with its original subprocess return code
and elapsed seconds. The corresponding `.txt` file is a byte-for-byte copy of
the complete stdout/stderr log, retained in a trackable extension because the
repository ignores `.log` files. A return code of zero for evaluation means the
evaluation executed successfully; it does not assert that the accuracy goal was
met. `target_achieved` and the separate acceptance check carry that conclusion.

The first training run (`train.json`) searched 36 candidates with outcome loss
weight 1. Its artifact `enhanced_head.pt` is historical and was never evaluated on
the holdout. After its weak training-only CV result, the final training run
(`train_rank_weights.json`) searched the same 36 kernel/regularization choices
with outcome weights 0.01, 0.1, and 1 (108 candidates total). All use the same
five task-disjoint folds and train-only normalization. The final search includes
the first search as a subset. These are tuning CV scores, not an unbiased nested
CV estimate of generalization. No holdout feature extraction or evaluation
occurred until after the final weight freeze.

Features consist of 349 observed temporal geometry summaries and 16,384 pooled
normalized embedding coordinates. Geometry includes norms, relative backward
changes, normalized state/action energy, direction changes, previous-action
alignment with the observed state change, and statistics/differences across
windows 1/4/12/32/full. Semantic features concatenate unit-normalized full and
last-12 state/action means. These are associations, not identified causal effects.
The final CV winner explicitly uses only the semantic kernel: geometry did not
win model selection. Geometry is exercised by the training ablations; it must
not be described as the source of a demonstrated improvement.

Training minimizes a weighted squared trajectory-outcome loss plus a
task-balanced squared positive-versus-negative margin loss and positive L2
regularization. The kernel eigenfeature system is solved directly; the saved
checkpoint contains the fitted coefficients, reference features, train-only
normalization, selected hyperparameters, CV results, split provenance, and source
hash. This trains a new trajectory ranking head; it does not fine-tune the 8B
backbone or the supplied baseline MLP. All 30,372 training step rows contribute
to pooled features for 298 trajectories, not 30,372 independent outcome targets.

No existing repository scoring module was replaced. The new CLI directly calls
feature extraction, training, frozen-model scoring, and reporting. Existing
unrelated code and baseline evidence are preserved.
