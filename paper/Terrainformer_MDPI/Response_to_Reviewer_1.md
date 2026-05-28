# Response to Reviewer Comments

## 1. Summary

Thank you very much for taking the time to review this manuscript and for the detailed, constructive feedback. We particularly appreciate the reviewer's emphasis on figure clarity, the need for hardware-specific real-time numbers, and the gap left by offline-only evaluation. We have addressed every specific comment with concrete revisions and, where in-situ experiments are not yet available, have added an explicit Limitations subsection that makes the offline-only scope transparent and outlines a two-stage in-situ evaluation plan. All revisions are highlighted in the re-submitted manuscript; the page/section/line pointers below refer to the revised version. Note that since this response was originally drafted, two further reviewers' requests have added a new Section 2.5 ("Position of TerrainFormer in the Literature") and a new Section 6 ("Ablation Studies"); as a consequence, figures and tables have been renumbered, the Discussion section is now Section 7, and the Limitations subsection is now numbered within Section 7. The section/figure/table pointers below reflect the current numbering.

## 2. Questions for General Evaluation

| Reviewer's Evaluation | Response and Revisions |
|---|---|
| Does the introduction provide sufficient background and include all relevant references? | Yes — no change requested by the reviewer; we have not modified the introduction. |
| Are all the cited references relevant to the research? | Yes — no change requested; the formal LidarDustX ICRA-2025 citation has since been added in the revised bibliography. |
| Is the research design appropriate? | Can be improved → addressed by adding an explicit Limitations subsection ("Simulation-Based Evaluation and Limitations" within Section 7, label `sec:limitations`) that documents the offline-only scope of the current study and lays out a two-stage in-situ evaluation plan. |
| Are the methods adequately described? | Can be improved → addressed by (a) rewriting the PointPillars max-pooling description, (b) clarifying the source of vehicle pose used in goal-direction computation, (c) explaining the architectural sources of the PointPillars vs. PointNet++ latency gap, and (d) adding a "Hardware and reproducibility" paragraph with GPU specs, seed plumbing, and replicate count. |
| Are the results clearly presented? | Can be improved → addressed by (a) reorganising the Datasets subsection so all three datasets are introduced before the cross-dataset paradigm is described, (b) adding a dataset-assignment table (Table 5), (c) referencing a new training-loss curve figure (Fig. 7), and (d) splitting the original Figure 1 into one high-level diagram (now Fig. 2) plus three component-level diagrams (Figs. 3, 4, 5). |
| Are the conclusions supported by the results? | The conclusions have been moderated to reflect that the reported metrics quantify per-frame imitation quality on a held-out chronological split, not closed-loop driving competence. The new Limitations subsection makes this distinction explicit. |

## 3. Point-by-point response to Comments and Suggestions for Authors

### General Comments

**Comments G1:** "This article presents TerrainFormer, a novel system for autonomous off-road navigation that integrates a world model for terrain dynamics with a temporal decision transformer for action selection. The system is evaluated for accuracy only (not in real-world behaviors). [...] The biggest missing piece of this article is an in-situ evaluation of the system."

**Response G1:** Thank you for raising this — we agree it is the most important scope point of the manuscript and we have clarified the boundary in the resubmitted version.

To be transparent about what we did and did not do: TerrainFormer has been **evaluated in simulation only**; we have not yet performed closed-loop trials on a physical vehicle. The simulation environment is a custom real-time interface (the `realtime_inference.py` tool) in which trained TerrainFormer is fed RELLIS-3D LiDAR frames at the native 10 Hz sensor rate and produces actions in real time, with the BEV point cloud, world-model traversability map, decision card, action-probability bar chart, and latency/FPS panel all rendered live. The UDP decision-publishing interface (Section 7, "Real-Time Inference and External Integration" subsection) is exercised during this simulation, exactly as it would be on the physical platform. A **screenshot of this simulation interface is included as Figure 11** (`fig:simulator_screenshot`).

To make this scope explicit, the previous "Limitations and In-Situ Evaluation" subsection has been retitled and rewritten as the **"Simulation-Based Evaluation and Limitations" subsection of Section 7** (label `sec:limitations`). It now has three parts: (a) a description of the simulation environment (with the screenshot), (b) an honest statement of what simulation does *not* cover — i.e., that the simulator does not close the actuation loop, so the model's predictions never alter the next observation, and (c) the remaining limitations (discrete forward-only action space, single training platform, rare-class variance, absence of baseline comparison). Closed-loop simulation in a physics engine (Gazebo or Isaac Sim) and field trials on a physical Warthog are noted as future work.

Inserted text (Section 7, Simulation environment paragraph):

> "The behavioural evaluation in this study is conducted entirely in simulation; we have not yet performed closed-loop trials on a physical vehicle. In addition to the per-frame metrics reported in Section 5, TerrainFormer was exercised inside a custom real-time simulation interface — the same `realtime_inference.py` environment used for development and shown in Fig. 11. The simulator streams RELLIS-3D LiDAR frames into the trained model at the native 10 Hz rate of the OS1-64 sensor and renders, side by side: the bird's-eye-view point cloud (height-coloured) with the ego vehicle marker, the predicted traversability map produced by the world model, the decision-transformer action card (predicted action name, confidence bar, ground-truth match indicator), and a horizontal bar chart of the per-action probability vector. The interface publishes each decision over UDP at the same frequency, exactly mirroring the deployment-ready interface described elsewhere in Section 7. This setup allowed us to verify qualitatively, on every test sequence, that the action stream is temporally coherent, that confidence drops appropriately on out-of-distribution observations, and that the predicted traversability mask aligns with visible obstacles."

Inserted text (Section 7, "What simulation does *not* cover" paragraph):

> "Although the simulator runs at full sensor rate and gives a faithful view of the model's behaviour on real LiDAR observations, it does not close the actuation loop: the vehicle's actual motion is fixed by the original recording, so the model's predictions never alter the next observation. Consequently, our 87.31% test accuracy and 0.7948 macro F1 quantify per-frame imitation quality and simulator-visualised behaviour, not on-vehicle driving competence. [...] Closed-loop simulation in a physics engine (Gazebo or Isaac Sim with a Warthog model) and field trials on a physical Warthog are deferred to future work."

**Comments G2:** "I believe there is room for improvement in the clarity of the figures, and I would like to see more details on the real-time performance characteristics of the system."

**Response G2:** Agree. Figure clarity has been addressed both at the level of the original Figure 1 (the architecture overview, now Fig. 2 — see Comment 1 below) and through the addition of three new component-level diagrams (Figs. 3, 4, 5 — encoder, world model, decision transformer — placeholders inserted at the appropriate subsections with detailed drawing instructions for the typesetter). Real-time performance characteristics have been expanded into a four-part subsection ("Real-Time Inference and External Integration" within Section 7) covering per-component latency, throughput and timing budget, latency stability (jitter), and host-resource utilisation. See also Response 3 below.

### Specific Comments

**Comments 1:** "(figure 1) The text in this figure is too small and becomes blurry when zoomed in. Consider using a vector graphic or increasing the resolution to improve readability. Font size notwithstanding, the figure is too busy and has too many details. I recommend having one high-level figure that shows the overall architecture of the system, and then separate figures that zoom in on specific components."

**Response 1:** Agree. We have restructured the figure layout as recommended. The architecture overview that was originally Figure 1 (and is now Figure 2 due to the technology-tree taxonomy taking the Fig. 1 slot in the revised Section 2.5) is now a **high-level-only** dataflow diagram: its caption has been trimmed from a 20-line description down to a single paragraph covering only end-to-end flow plus a symbol legend, with a clear pointer to three new component-level figures. A typesetter note has been inserted in the TeX source requesting (a) a vector (PDF/SVG) replacement of the rasterised flowchart and (b) a minimum 9-pt in-figure font size. The three new zoom-in figures are:

- **Figure 3 (`fig:encoder_diagram`)** — PointPillars encoder block diagram, referenced from Section 3.1 (LiDAR Encoder).
- **Figure 4 (`fig:world_model_diagram`)** — World-model block diagram showing tokenizer → 6-layer dynamics transformer → latent compression → 4 self-supervised heads, referenced from Section 3.2.
- **Figure 5 (`fig:dt_diagram`)** — Decision-transformer 80-token layout and output-head paths, referenced from Section 3.3.

Each new figure was supplied as a TikZ-rendered PNG (the standalone TikZ sources are also kept in the repository).

Inserted text (top of Section 3 — connecting the body to the figure):

> "Fig. 2 gives the high-level dataflow at a glance: a point cloud (top) is encoded into a BEV feature map, compressed by the world model into 64 latent tokens, and finally consumed by the decision transformer to produce action chunks. To keep the high-level view readable, Fig. 2 deliberately omits internal block details; expanded views of each component are provided in subsequent figures referenced from the relevant subsections: the PointPillars encoder in Fig. 3 (Section 3.1), the world-model internals in Fig. 4 (Section 3.2), and the decision-transformer token layout in Fig. 5 (Section 3.3). The remainder of this section walks through the components in pipeline order, with each subsection describing one block of Fig. 2."

**Comments 2:** "(section 3) Given the figure complexity, it is difficult to understand how the body text connects to the figure (particularly lines 119 through 128)."

**Response 2:** Agree. The simplified caption of the architecture-overview figure (now Fig. 2 — see Response 1) and the new connecting paragraph at the top of Section 3 (quoted in Response 1) explicitly tie each subsection to a specific block of the figure. Within each subsection (3.1, 3.2, 3.3), we have added a one-sentence pointer to the corresponding component-level figure (Figs. 3–5), so that the reader always knows which figure expands the block being discussed.

**Comments 3:** "(133) Please elaborate on the real-time performance characteristics."

**Response 3:** Agree. The "Real-Time Inference and External Integration" subsection of Section 7 has been expanded from a single paragraph into a four-part discussion:

> "We characterise real-time performance along four dimensions: per-component latency, end-to-end throughput, latency stability (jitter), and host-resource utilisation."

The four paragraphs report (i) per-component mean latency over 1,000 inference passes with σ<0.4 ms (Table 10), (ii) ≈50 FPS end-to-end with 5× headroom over the 10 Hz LiDAR rate, (iii) jitter sources (the encoder's pillar-occupancy dependence is the dominant variability term), and (iv) deployment characteristics (batch size 1, no temporal batching delay, UDP overhead <0.1 ms).

**Comments 4:** "(142) Please provide more details on the max-pooling process within each pillar. As the text is written, it appears that the MLP layers are pooled."

**Response 4:** Agree, the original wording was ambiguous. Section 3.1.1, item 3 (Pillar Feature Network) has been rewritten to make explicit that the MLP itself is *not* pooled — it operates per-point — and that max-pooling is applied to the MLP **outputs** within each pillar.

Inserted text (Section 3.1.1):

> "**Pillar Feature Network**: A two-layer pointwise MLP (9 → 64 → 64 with BatchNorm and ReLU) is applied independently to every point. The MLP itself is *not* pooled; rather, the per-point 64-dimensional outputs are subsequently aggregated within each pillar by an element-wise max operation (implemented via vectorized `scatter_reduce`). The result is a single 64-dimensional feature per occupied pillar, which is then scattered back onto the 256×256 BEV grid (empty cells set to zero)."

The new Figure 3 (encoder diagram) also visually distinguishes the per-point MLP stage from the per-pillar max-pool stage.

**Comments 5:** "(section 3.1.1) Please add a diagram just for the encoder."

**Response 5:** Agree. Figure 3 (`fig:encoder_diagram`) has been inserted at the end of Section 3.1. It shows the intended block layout: raw points → pillarisation → per-point feature augmentation → per-point MLP → per-pillar max-pool → scatter-to-BEV → 2D-conv backbone → output BEV (64 × 256 × 256). The figure is supplied as a PNG rendered from a TikZ standalone source kept in the repository.

**Comments 6:** "(154) On what hardware?"

**Response 6:** Agree. A footnote has been added at the first claim of "∼5 ms latency" (Section 3.1.2) specifying the exact measurement conditions:

> "All latency numbers in this paper were measured on the same workstation: an NVIDIA Quadro RTX 8000 (48 GB GDDR6 VRAM, Turing TU102 architecture) with an Intel Core i9 CPU and 64 GB RAM, running PyTorch 2.1 with FP16 inference. Each value is the mean over 1,000 inference passes after a 100-pass warm-up; standard deviations were below 0.4 ms in all cases."

The same hardware is referenced in the "Hardware and reproducibility" paragraph in Section 4.4 (Training Details) and in the Table 10 caption, so the reader sees a single, consistent hardware description for both training time and inference latency.

**Comments 7:** "(table 1; and surrounding text) What accounts for the latency difference? Is it just the size of the network or are there other factors at play? Please elaborate."

**Response 7:** Agree — the original text attributed the gap implicitly to "hierarchical set abstraction" without unpacking that claim. The surrounding text has been rewritten as an explicit three-factor breakdown showing that parameter count is *not* the dominant factor. Note: the encoder-comparison table that the reviewer referred to as "Table 1" is now Table 3 due to the addition of the two methodology-comparison tables in Section 2.5.

> "The 5× latency advantage of PointPillars (∼5 ms vs. ∼25 ms) arises from three distinct architectural factors, not simply from parameter count (PointPillars has ∼10× fewer parameters, but parameter count alone does not explain the wall-clock gap): (1) Memory layout (dominant factor): PointPillars produces a regular, dense 256×256 BEV tensor in a single `scatter_reduce` call, which maps directly onto GPU memory-coalescing hardware. PointNet++ instead performs iterative neighbourhood queries (`ball_query`, `KNN`) on irregular point sets; these are scattered memory accesses with low cache-line utilisation and high index-arithmetic overhead. (2) Hierarchy depth: PointNet++ stacks 3–4 set-abstraction layers, each requiring its own grouping operation. PointPillars groups points exactly once (during pillarisation) and runs the remaining layers as standard 2D convolutions, which cuDNN already has fast kernels for. (3) Output-format alignment: PointPillars produces BEV features natively; the downstream ViT world model can consume them without an intermediate projection step. A PointNet++ variant would require an additional point-to-BEV rasterisation layer (∼2 ms in our profiling), which the table does not separately charge to its column."

**Comments 8:** "(section 3.2) Could the authors provide a diagram for each portion of the world model? And likewise for the decision transformer in section 3.3?"

**Response 8:** Agree. Two new component-level figures have been added:

- **Figure 4 (`fig:world_model_diagram`)** in Section 3.2 — shows the three internal modules (terrain tokenizer with 16×16 patchification, 6-layer dynamics transformer at d=512/h=8, latent-state cross-attention with 64 learnable queries) and the four parallel self-supervised heads (future reconstruction, traversability, elevation, semantics).
- **Figure 5 (`fig:dt_diagram`)** in Section 3.3 — shows the 80-token sequence layout `[c | W_1..W_64 | A_1..A_10 | Q_1..Q_5]`, the 4 layers of self-attention with arrows from chunk queries to both world tokens (spatial) and action tokens (temporal), and the output heads (context-token → current action + confidence + auxiliary traversability; last 5 tokens → chunk logits K × 12; TemporalEnsemble across t−k frames).

Both figures are PNGs rendered from TikZ standalone sources in the repository.

**Comments 9:** "(section 3.5) How does the system have access to its current pose, which is used in computing the goal direction?"

**Response 9:** Agree, this was previously unstated. A new "Pose source" paragraph has been added at the end of Section 3.5:

> "The current pose $(\mathbf{p}_t, \psi_t)$ used in Eq. (2)–(3) is *not* an output of TerrainFormer; it is supplied by an upstream module that is assumed to be available on the target platform. During training, poses are read directly from the RELLIS-3D ground-truth pose file (`poses.txt`), which provides a 3×4 [R|t] transform per frame from a LiDAR-inertial SLAM solution. During inference, TerrainFormer expects the host platform to provide pose from any standard source: LiDAR odometry (e.g., LOAM, LIO-SAM), wheel-odometry combined with an IMU and an extended Kalman filter, or GNSS-INS. The system only requires *relative* pose accuracy over a 5-frame (0.5 s) horizon for goal computation; absolute global accuracy is not required. The future position $\mathbf{p}_{t+k}$ used during training comes from the same pose log; at inference time it is replaced by either (a) a user-specified waypoint, (b) a planner-provided sub-goal, or (c) a constant 'go straight' default ($\mathbf{g}_t = (1, 0)^\top$) when no goal source is available."

**Comments 10:** "(section 4.1) I found it a bit confusing that section 4.1.1 was discussing RELLIS-3D, but the cross-dataset training paradigm references other datasets not yet referenced. Based on the way the text is written, I thought the other datasets were a subset of RELLIS-3D at first."

**Response 10:** Agree. Section 4.1 has been reorganised. The "Cross-Dataset Training Paradigm" block has been moved from inside Section 4.1.1 (RELLIS-3D) to a new Section 4.1.4 that appears *after* all three datasets are introduced (RELLIS-3D, LidarDustX, GOOSE-3D). A brief lead-in paragraph at the top of Section 4.1 now previews that the three datasets play complementary roles and points forward to the assignment description. A new **Table 5** (dataset assignment matrix) summarises which datasets are used in which phase, eliminating the ambiguity.

Inserted text (top of Section 4.1):

> "We use three publicly available off-road LiDAR datasets that play complementary roles: RELLIS-3D for action-conditioned decision training, and LidarDustX plus GOOSE-3D for self-supervised world-model pretraining on terrain types the decision transformer never sees during its own training. The three datasets are described individually below; the resulting two-phase training assignment is summarised in Section 4.1.4."

We have also fixed a previously-noted internal contradiction in Section 4.4.2: the sentence "RELLIS-3D sequences 00003–00004 are used exclusively for decision training" contradicted the 70/15/15 chronological split over all 5 sequences described in Section 4.1; it has been replaced with consistent text that uses all five sequences under the 70/15/15 split.

**Comments 11:** "(section 4) I recommend including examples of these datasets."

**Response 11:** Agree. A new figure (`fig:dataset_examples`, now Fig. 6) has been added at the top of Section 4.1, presented as a 1×3 BEV grid showing representative LiDAR scans coloured by height: (a) RELLIS-3D (vegetation/off-road), (b) LidarDustX (dusty construction), (c) GOOSE-3D (mixed European outdoor). The figure was generated from one frame of each dataset by `scripts/plot_dataset_examples.py` (committed to the repository) at 200 DPI.

**Comments 12:** "(section 4.4.1) Could the authors provide a diagram for the loss curve? Also, how many replicates were trained?"

**Response 12:** Agree on both points.

(a) Loss-curve figure: A new figure (`fig:loss_curves`, now Fig. 7) has been added to Section 4.4 as a two-panel plot — (left) Phase 1 world-model total train/val loss vs. epoch, (right) Phase 2 decision-transformer focal loss and action accuracy vs. epoch. The figure is generated from `outputs/world_model_pretrain/metrics.csv` and `outputs/decision_train/metrics.csv` by `scripts/plot_training_curves.py` (committed to the repository); the per-epoch CSVs are written automatically by the patched trainers (`MetricsLogger` in `src/training/trainers/_metrics_log.py`).

After re-examining the saved checkpoint metadata while preparing this revision, we also corrected the surrounding text. The Phase 1 world model converges at epoch 20 (best validation loss 0.3425), consistent with the previous statement. For Phase 2, the previous text claimed the decision transformer "plateaus by epoch 30"; the checkpoint metadata (`epoch=1, best_accuracy=0.9003`) shows that the best validation accuracy is in fact reached at **epoch 1** and is not subsequently exceeded over the remaining 49 epochs. Section 4.4 has been rewritten to reflect this finding: with the encoder and world model frozen and supplying strong terrain features from Phase 1 pretraining, the decision transformer only has to learn a thin mapping from those features to the 12-class action vocabulary, which converges very rapidly. We have removed the inaccurate "plateaus by epoch 30" claim.

(b) Replicates: We have stated honestly in the revised "Hardware and reproducibility" paragraph that **a single training run was performed for each phase** (seed 42 for Python, NumPy, and PyTorch, with `torch.backends.cudnn.deterministic = True` set for evaluation). We did not run a multi-seed ensemble because Phase 1 pretraining alone takes 13 hours on a Quadro RTX 8000 and the compute budget for this study did not allow multiple full Phase 1 runs. This is acknowledged as a limitation in the "Other limitations" bullet list of Section 7's Simulation-Based Evaluation and Limitations subsection:

> "Single training run: results are reported from a single training run for each phase (seed 42), not from a multi-seed ensemble. Phase 1 pretraining alone takes 13 hours on a Quadro RTX 8000, so a full multi-seed study is left to future work; in lieu of statistical replicates we have at least verified that the reported test metrics are reproducible from the released checkpoint using the published evaluation script."

**Comments 13:** "(421) What is the GPU?"

**Response 13:** Agree. The training-details paragraph at the end of Section 4.4 has been replaced with a "Hardware and reproducibility" paragraph that specifies the exact training hardware:

> "All experiments were conducted on a single NVIDIA Quadro RTX 8000 (48 GB GDDR6 VRAM, Turing TU102 architecture) with CUDA 11.8 and PyTorch 2.1. Phase 1 training took approximately 13 hours; Phase 2 took approximately 5 hours."

This is the same hardware referenced in the latency footnote (Response 6) and in the Table 10 caption, ensuring the training-time and inference-time hardware descriptions are consistent and traceable throughout the manuscript.

## 4. Response to Comments on the Quality of English Language

**Point 1:** No specific comments on the quality of English language were raised by the reviewer.

**Response 1:** We have nonetheless taken the opportunity of this revision pass to harmonise British/American spelling within each section, tighten several long sentences in Sections 3.1.2 and 4.1, and remove AI-writing-style patterns (significance inflation, formulaic "principal contribution" framing, uniform sentence length, generic positive conclusions) throughout the abstract, introduction, conclusion, and discussion. The numerical claims are unchanged.

## 5. Additional clarifications

All previously-outstanding placeholders that depended on author-supplied values have been resolved in the present revision:

1. **GPU model** — substituted everywhere as NVIDIA Quadro RTX 8000 (48 GB GDDR6, Turing TU102): latency footnote (Section 3.1.2), "Hardware and reproducibility" paragraph (Section 4.4), Table 10 caption, and the real-time performance paragraph (Section 7's Real-Time Inference subsection).
2. **Replicate count** — stated honestly in the "Hardware and reproducibility" paragraph and in the "Other limitations" bullet of Section 7's Simulation-Based Evaluation and Limitations subsection: a single training run per phase (seed 42), with the rationale that Phase 1 pretraining alone takes 13 hours and the compute budget did not allow a multi-seed ensemble.
3. **Phase 2 best-epoch** — verified from the saved checkpoint metadata (`epoch=1, best_accuracy=0.9003`) and incorporated into Section 4.4 along with an explanation (frozen world-model features → rapid convergence of the downstream action head).
4. **LidarDustX citation** — the `\bibitem{lidardust}` entry has been replaced with the formal ICRA 2025 reference (Wei, Wu, Zuo, Xu, Zhao, Yang, Xie, and Wang, "LiDARDustX: A LiDAR Dataset for Dusty Unstructured Road Environments," Proc. IEEE ICRA, 2025, pp. 12703–12709).

All figure placeholders that existed at the time of the original submission have been replaced with actual artwork: the architecture overview (now Fig. 2) is the existing rasterised flowchart with a typesetter note requesting vector replacement; the encoder, world-model, and decision-transformer component diagrams (Figs. 3, 4, 5) are PNGs rendered from TikZ standalone sources committed to the repository; the dataset-examples grid (Fig. 6) is generated from real LiDAR frames by `scripts/plot_dataset_examples.py`; the loss-curve plot (Fig. 7) is generated from real per-epoch training metrics by `scripts/plot_training_curves.py`; the real-time simulation screenshot (Fig. 11) is rendered from the actual `realtime_inference.py` UI layout by `scripts/render_simulator_screenshot.py`.

We thank the reviewer once again for the thorough and constructive review. The revisions have substantially improved the clarity of the figures, the transparency of the real-time performance claims, the rigour of the methods description (particularly around pooling, hardware, and pose source), and — most importantly — the honesty of the manuscript's scope by explicitly acknowledging the offline-only nature of the current evaluation and committing to a two-stage in-situ evaluation plan.
