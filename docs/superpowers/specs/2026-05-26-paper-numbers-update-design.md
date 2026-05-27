# Paper Numbers Update — Design Doc

**Date**: 2026-05-26
**Author**: rickslab3 (via brainstorming skill)
**Status**: design, pending implementation
**Sub-project**: A of 4 (A → C → B → D — see top-level decomposition in conversation context)
**Estimated wall-clock**: ~3 hours (2 h GPU retrain in background + ~1 h text edits + verification)

## 1. Motivation

The MDPI paper draft ([paper/Terrainformer_MDPI/terrainformer_mdpi.tex](../../../paper/Terrainformer_MDPI/terrainformer_mdpi.tex)) reports results from an earlier training run whose `actions.npy` label files no longer exist on disk. When we re-ran the full pipeline on 2026-05-23/24 the auto-generated action labels (via `scripts/generate_action_labels.py` from poses) produced a different action distribution, giving:

| Metric | Paper (old) | New run | Δ |
|---|---:|---:|---:|
| Test accuracy | 87.31 % | 84.65 % | −2.66 |
| Macro F1 | 0.7948 | 0.5323 | −0.26 |
| Predictive accuracy drop | 0.79 % | 0.00 % | +0.79 |
| Predictive agreement | 98.82 % | 100.00 % | +1.18 |

Two cells of the paper's per-class Table 5 were also internally inconsistent with the paper's action-space definition in Section 3.4 (IDs 7 and 9 reversed). Re-running the pipeline surfaced this as a true bug in `scripts/generate_action_labels.py` (current code: 7 = R_slight, 9 = R_sharp; paper's intended convention from Table 5: 7 = R_sharp, 9 = R_slight).

User decision (this brainstorming session): **the new training run is the truth**. We update the paper, the code-vs-paper label mismatch is fixed in the code (not the paper), and the Phase 2 model is retrained with corrected labels so the integer values in `actions.npy` physically agree with their advertised names.

## 2. Scope

In scope:
1. Swap the integer-to-class mapping for IDs 7 and 9 in code (`generate_action_labels.py`, `decision_trainer.py` ACTION_NAMES, `evaluate_decision.py`, `label_actions.py`)
2. Regenerate `actions.npy` + `goals.npy` for sequences 00000–00004
3. Re-run Phase 2 training (~2 h on Quadro RTX 8000)
4. Re-run `evaluate_decision.py` and `evaluate_predictive.py`
5. Update the four user-selected files with the final new numbers:
   - [paper/Terrainformer_MDPI/terrainformer_mdpi.tex](../../../paper/Terrainformer_MDPI/terrainformer_mdpi.tex)
   - [paper/terrainformer_flowchart.tex](../../../paper/terrainformer_flowchart.tex)
   - [README.md](../../../README.md)
   - [paper/Terrainformer_MDPI/Response_to_Reviewer.md](../../../paper/Terrainformer_MDPI/Response_to_Reviewer.md) (+ regenerate `.docx`)
6. Update Section 6 of the .tex prose where it makes claims about per-class F1, paragraph-level (see § 5 below)
7. Update abstract, conclusions, fig 6 caption to be consistent with the new headline numbers
8. Recompile the .pdf
9. Verify no stale numbers remain via `grep`

Out of scope:
- `paper/terrainformer.tex` (the older, non-MDPI version)
- Phase 1 retraining (Phase 1 doesn't use action labels; results unchanged)
- Architectural / methodology changes
- Figure artwork generation (handled by sub-project B)
- Code cleanup of trainer CSV hook etc. (sub-project C)
- Final code review (sub-project D)

## 3. Approach

**Approach 1: manual surgical edits** (selected from three options).

Selected over (2) script-driven mapping and (3) delegate-to-agent because:
- Section 6 prose changes are not number-replacements — they need direction-flipping ("classes 8/9 collapse" instead of "classes 8/9 match dominant-class performance")
- Manual editing produces a clean diff that sub-project D (code-review) can audit
- Lowest hallucination risk

## 4. Code change (label swap)

After audit, exactly four files contain the 7 ↔ 9 mapping:

[scripts/generate_action_labels.py](../../../scripts/generate_action_labels.py) — four edits:
- Line 31/33: swap `7: 'TURN_RIGHT_SLIGHT'` ↔ `9: 'TURN_RIGHT_SHARP'` in the `ACTIONS` dict
- Lines 250/252: swap the same names in the docstring of `discretize_actions`
- Line 294: change `actions[i] = 4 if omega > 0 else 9` → `else 7` (sharp turn negative-omega branch)
- Line 300: change `actions[i] = 6 if omega > 0 else 7` → `else 9` (slight turn negative-omega branch)

After this change, a sharp-right physical motion is written as integer 7 in `actions.npy`, a slight-right motion as integer 9. This matches the paper's Table 5 convention.

[src/training/trainers/decision_trainer.py](../../../src/training/trainers/decision_trainer.py) — swap entries for keys 7 and 9 in the `ACTION_NAMES` dict at line 16 so eval prints match.

[scripts/evaluate_decision.py](../../../scripts/evaluate_decision.py) — same `ACTION_NAMES` swap at line 30.

[src/models/decision/action_tokenizer.py](../../../src/models/decision/action_tokenizer.py) — swap `TURN_RIGHT_SLIGHT = 7` and `TURN_RIGHT_SHARP = 9` at lines 35 and 37 of the enum. This is documentation-level (the enum is imported as a class but no code references the names directly), but keeping it in sync prevents future confusion.

`scripts/label_actions.py` and `scripts/decision_receiver.py` do not need changes — `label_actions.py` uses no per-ID dict in this range; `decision_receiver.py` uses a different (18-action legacy) mapping that's unrelated to the 12-action paper convention.

Regeneration: `python -c "<call generate_action_labels_for_sequence for 5 seqs>"` (~1 min).

Retraining: `python scripts/train.py --config configs/training/train_decision.yaml --device cuda` (~2 h).

Re-evaluation: `python scripts/evaluate_decision.py --data /home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/` then `python scripts/evaluate_predictive.py --data /home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/`.

## 5. Discussion-prose changes (Section 6 of .tex)

| § | Current claim | New claim |
|---|---|---|
| 6.1 Cross-dataset generalization | Unchanged. Phase 1 reproduced cleanly; the cross-dataset claim still holds. |
| 6.2 Two-phase pipeline | Unchanged. Architecture facts are stable. |
| 6.3 Class imbalance | **Rewrite**. Current: "All 12 classes achieve F1 ≥ 0.65". New (template; exact counts depend on retrain output): "Of the well-represented classes (support > 30), the model achieves F1 ≥ 0.65 on stop, fwd_fast, fwd_slow, and L_slight. The remaining minority classes — particularly the medium / sharp right turns and the combined fwd_L / fwd_R actions — remain heavily affected by imbalance despite focal loss with inverse-frequency weighting. The combined-action classes (fwd_L n≈2, fwd_R n≈0 under our auto-generated labels) are at or below the threshold of evaluability and we report their F1 as nominal-zero rather than learned." Add one sentence: "Focal loss and inverse-frequency weighting are insufficient when minority class support drops below ~10 test samples; resolving this would require either denser action labels per sequence or a hierarchical action prediction head." |
| 6.4 Val-test gap | Update numbers only: 90.03 → 89.53, 87.31 → 84.65. Conclusion (mild distribution shift) unchanged. |
| 6.5 Multi-sensor WM | Unchanged. |
| 6.6 Real-time inference | Unchanged (latency claims are independent of action labels). |
| 6.7 Limitations | Unchanged — the "single training run" honesty bullet is already there. |
| Sec 7 Conclusion | Update headline numbers + soften "all 12 classes achieve F1 ≥ 0.65" claim. |
| Abstract | "87.31 % test accuracy with 0.7948 macro F1" → "84.65 % test accuracy with 0.5323 macro F1". "0.79 % accuracy drop, 98.82 % agreement" → "0.00 % accuracy difference, 100 % action agreement (predictive eval is functionally equivalent to ground-truth observation)". |
| Fig 6 caption | Update best Phase 2 val/accuracy figure (currently 0.8953; will change slightly after retrain). |

## 6. Verification

After all edits:

```bash
# 1. No stale numbers anywhere we touched
grep -nE "87\.31|0\.7948|0\.7828|0\.8127|90\.03|86\.52|98\.82" \
    paper/Terrainformer_MDPI/terrainformer_mdpi.tex \
    paper/terrainformer_flowchart.tex \
    README.md \
    paper/Terrainformer_MDPI/Response_to_Reviewer.md \
  && echo "ERROR: stale numbers remain" \
  || echo "OK: no stale numbers"

# 2. Per-class table cross-check: support counts sum to 2033
# (manual once after retrain)

# 3. Macro F1 = mean of per-class F1 (within rounding)
# (computed from the new evaluation_results.txt)

# 4. PDF rebuilds
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$(pwd)":/work -w /work texlive/texlive:latest \
    bash -c "for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error terrainformer_mdpi.tex; done"

# 5. Use verification-before-completion skill at the end
```

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Retrain produces materially different numbers from the current 84.65 / 0.5323 | Accept whatever it produces; do not iterate. Note the retrain in commit message. |
| `discretize_actions` has more than one place that emits 7 or 9 → easy to miss one | Read the full function before editing; add a unit-style sanity check in implementation plan (e.g. assert `ACTIONS[7] == 'TURN_RIGHT_SHARP'` after the swap). |
| Stale paper sections claim "all 12 classes F1 ≥ 0.65" — easy to miss one occurrence | `grep -n "0\.65"` across the .tex as part of verification step. |
| Abstract talks about "98.82 %" — if not updated everywhere, reviewers will catch inconsistency | Item 1 of verification covers this. |
| Recompile breaks due to changed table sizes (extra column not fitting) | Verify PDF compiles before declaring done. |
