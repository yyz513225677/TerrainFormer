# Response to Reviewer 3 Comments

## 1. Summary

Thank you for the careful review and for raising five substantive points that all sharpen how the manuscript's scope and limitations should be presented. We respond to each comment individually below. Where the comment matches content that is already present in the revised manuscript we point to the relevant section; where it identifies a genuine gap we describe the change we will make in the next revision and, when applicable, the reasons we have not yet been able to run the requested experiment.

## 2. Questions for General Evaluation

| Reviewer's Evaluation | Response and Revisions |
|---|---|
| Does the introduction provide sufficient background and include all relevant references? | Can be improved → reviewer 3 has identified an additional sensor-fusion reference \cite{star_inertial_fusion} that we will add to Section 2 (Related Work) and discuss in the context of degraded-perception robustness (Comment 2). |
| Are all the cited references relevant to the research? | Yes — with the addition of the inertial / star-sensor fusion reference suggested by reviewer 3 (Comment 2). |
| Is the research design appropriate? | Yes; the cross-dataset training protocol, real-time inference budget, and discrete-action formulation are appropriate for the stated off-road off-line behavioural-cloning objective. The honest scope is documented in Section 7 (Limitations) and reinforced by reviewer 3's Comments 4 and 5. |
| Are the methods adequately described? | Yes — Section 6 (Ablation Studies) and Section 3.5 (Pose source) document the methodological choices reviewer 3 asks about. We will extend the relevant subsections per the responses below. |
| Are the results clearly presented? | The results are clearly presented for what is currently measured (offline classification on held-out RELLIS-3D). Reviewer 3 is correct that closed-loop safety, dynamic obstacle response, and sensor-degradation robustness are not yet measured; this is honestly flagged in Section 7. |
| Are the conclusions supported by the results? | Yes for the in-scope claims (per-frame imitation quality on the chronological split, cross-dataset transfer of the world model). The claims that fall outside this scope (closed-loop driving competence, real-platform deployment) are not made; Section 7 explicitly marks them as future work. |

## 3. Point-by-point response to Comments and Suggestions for Authors

### Comments 1: Component-wise decomposition of the 0.79\% predictive degradation

> "The proposed TerrainFormer architecture is technically comprehensive, but the manuscript lacks sufficient ablation analysis on the contribution of the world model, action chunking, and TemporalEnsemble modules. In particular, the reported 0.79\% predictive degradation should be decomposed component-wise to clarify which architectural elements dominate cross-dataset generalization performance."

**Response 1:** The reviewer is correct. The 0.79\% predictive-degradation number (Table 6 in Section 5.4) is a *summary* metric: it measures the action-selection sensitivity when the world model's predicted future frame substitutes for the ground-truth observation, but it does not isolate the individual contributions of (a) the world model itself, (b) action chunking with TemporalEnsemble, and (c) the cross-dataset training protocol. Section 6 (Ablation Studies) tags these three items honestly: §6.2 (predictive evaluation) is empirical; §6.4 (action-chunk size) and the implied TemporalEnsemble contribution are reported as design rationale, not measured ablations.

We propose to address this in the next revision in two steps:

1. **Empirical decomposition (planned)**: run four additional Phase 2 retrains, each with one component removed: (i) world model off (decision transformer reads raw BEV features), (ii) action chunking off ($K{=}1$, no TemporalEnsemble), (iii) TemporalEnsemble off ($K{=}5$ but argmax-per-frame instead of weighted average), and (iv) cross-dataset protocol off (Phase 1 also trained on RELLIS-3D). Each retrain is approximately 2 hours on a Quadro RTX 8000; the total compute is approximately 8 hours. Results would form a new subsection 6.3a "Component-wise decomposition" within the existing Ablation Studies section.

2. **Interim qualitative decomposition**: in the meantime, we will expand Section 6.2 to add the following decomposition argument from the existing measurements: the 0.79\% drop comprises (i) world-model prediction error introduced when its 5-frame future BEV reconstruction deviates from the ground truth, and (ii) the policy's robustness to that deviation, which TemporalEnsemble's weighted average across overlapping chunks partially absorbs. The 98.82\% agreement rate already bounds the policy-robustness contribution from below, since most predictions are unchanged regardless of which frame is fed in.

We will defer the empirical decomposition (item 1) to the next revision pass and clearly mark the planned experiments in Section 6.

### Comments 2: Sensor fusion under degraded perception (inertial / star-sensor reference)

> "The navigation framework would benefit from stronger discussion of integrated inertial sensing robustness under degraded perception conditions. The authors should add some discussions about navigations, such as 'Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method,' particularly regarding sensor fusion reliability, alignment stability, and fault-tolerant navigation initialization in complex environments."

**Response 2:** Acknowledged. Section 3.5 (Goal Direction Computation) currently lists LiDAR odometry, wheel-odometry + IMU + EKF, and GNSS-INS as acceptable pose sources at inference time, but does not develop the robustness-under-degraded-perception argument. We will:

1. Add the suggested reference \cite{star_inertial_fusion} to the bibliography and cite it in Section 3.5 when introducing IMU-based fallback sources.
2. Extend the "Pose source" paragraph in Section 3.5 with a short discussion of three points the cited work emphasises: (a) the fusion-reliability argument that inertial sensing provides a continuous pose signal during temporary failure of the primary attitude sensor (in our setting, LiDAR-derived odometry under heavy dust, fog, or vegetation occlusion); (b) the alignment-stability requirement that the IMU's bias and scale-factor calibration must be tracked to keep the integrated attitude useful over the 0.5-second goal-prediction horizon TerrainFormer uses; and (c) the fault-tolerant initialisation pattern that allows the navigation stack to recover after a brief sensor outage without restarting from a cold initial estimate.
3. Add a sentence in Section 7 (Limitations) noting that degraded-perception robustness is documented at the architectural level only; empirical evaluation under controlled sensor-degradation conditions is deferred to future work and will be reported alongside the closed-loop trials (Comment 5).

Planned insertion (Section 3.5, end of the "Pose source" paragraph):

> "When LiDAR-based pose estimation is temporarily unavailable (e.g., heavy dust or fog), TerrainFormer's goal-direction input falls back to inertial-only integration. The principles described in the inertial / star-sensor fusion literature \cite{star_inertial_fusion} are directly relevant here: a continuously running IMU integrator provides the relative-pose signal needed for goal-direction computation during primary-sensor outage, while bias and scale-factor estimates from the EKF maintain attitude stability over the 5-frame (0.5-second) goal-prediction horizon. We treat these robustness aspects as deployment-platform concerns rather than network-architecture concerns; their empirical evaluation is part of the closed-loop work outlined in Section 7."

### Comments 3: Cooperative motion stability and multi-step control under rapid terrain changes

> "Although the decision transformer demonstrates promising trajectory consistency, the manuscript does not sufficiently analyze cooperative motion stability and dynamic interaction during complex maneuvers. The authors should discuss connections with temporal smoothness constraints, anticipatory trajectory adaptation, and multi-step control stability under rapidly changing terrain conditions."

**Response 3:** This is a fair characterisation. The current manuscript discusses TemporalEnsemble (Section 3.3 and the planned Section 6.4) as the mechanism for output-stream smoothness, but does not connect it explicitly to the three concepts reviewer 3 lists: temporal smoothness constraints, anticipatory trajectory adaptation, and multi-step control stability under rapid terrain changes.

We will:

1. **Expand Section 3.3 (Decision Transformer) with a paragraph on temporal smoothness constraints.** TemporalEnsemble's exponential-decay weighting ($\lambda{=}0.9$) is precisely a soft temporal-smoothness prior over the action stream: a sudden disagreement between the current frame's chunk prediction and the four previous frames' overlapping predictions is averaged out, while a consistent action change (e.g., a real turn) is preserved because all overlapping chunks agree.

2. **Add a paragraph to Section 6.4 (Chunk size $K$) on anticipatory trajectory adaptation.** With $K{=}5$ and 10\,Hz LiDAR, the decision transformer predicts the next 0.5 seconds of actions per frame. When the terrain in front of the vehicle changes (e.g., a transition from packed dirt to soft mud appears in the BEV traversability map), the chunk predictions in subsequent frames adapt to the new terrain class before the vehicle physically reaches it; TemporalEnsemble's weighted aggregation transitions the output stream smoothly across the discontinuity rather than producing a single hard frame-to-frame switch.

3. **Add a paragraph to Section 7 (Limitations) on multi-step control stability.** Open-loop evaluation (Section 7) cannot exhibit divergence from the recorded trajectory, so the manuscript cannot currently report the multi-step control-stability metrics (cross-track error growth rate, attitude oscillation amplitude, action-stream cross-correlation under terrain-class boundary events) that reviewer 3 asks about. These metrics are part of the closed-loop simulation plan outlined in Section 7; we will report them in the next revision once the simulation-in-the-loop pass is complete.

### Comments 4: Sensor precision and navigation drift impact on decision quality

> "The manuscript emphasizes real-time off-road navigation accuracy, yet the influence of sensor precision and navigation drift on downstream decision quality is not rigorously analyzed. The authors need to add some analyses about this part, particularly regarding high-precision attitude stabilization and its implications for robust autonomous navigation performance."

**Response 4:** Acknowledged. The current analysis assumes ground-truth pose information from the RELLIS-3D SLAM-derived `poses.txt` file. Two sensor-precision-related effects are therefore not measured: (a) the impact of LiDAR sensor noise (intensity dropout, ring patterns, motion-blur artefacts at high vehicle speed) on the world model's latent terrain representation, and (b) the impact of accumulated pose drift on the goal-direction computation that the decision transformer uses as a conditioning signal.

We will:

1. **Add a robustness-analysis subsection to Section 7 (Limitations)** explaining both effects. For (a), the PointPillars encoder is relatively robust to per-point noise because the per-pillar max-pool reduces dependence on individual point measurements, but it is sensitive to systematic ring-pattern artefacts that can mimic terrain edges; the world model's traversability head is the most exposed component because it is directly supervised by a geometric proxy. For (b), the goal-direction input is computed from a 5-frame future position, so absolute global drift does not affect it (only relative pose stability over 0.5 seconds matters); the IMU-fallback path described in Response 2 is the principal mitigation for short-term degradation of the LiDAR-based pose stream.

2. **Add the planned attitude-stability experiment to Section 7's future-work list.** A controlled study where artificial IMU noise is injected into the goal-direction computation, and the resulting downstream decision-stream stability is measured against the noise-free baseline, would directly address reviewer 3's concern. This experiment is offline (no retraining required); only inference needs to be re-run with corrupted pose input. We estimate approximately 2 hours of work to set up and report.

Planned insertion (Section 7, new "Robustness analysis" subsection):

> "Two sensor-precision effects are not measured in the present study: per-point LiDAR noise propagating through the world model, and accumulated pose drift affecting the goal-direction input. The PointPillars encoder's per-pillar max-pool absorbs the first effect by construction; the second is bounded by the fact that goal direction is computed from a 5-frame relative displacement (0.5 s horizon), making absolute global drift irrelevant. We outline a controlled pose-noise injection experiment in the next-revision plan and reference the inertial-fusion attitude-stabilisation principles in \cite{star_inertial_fusion} as the deployment-side mitigation."

### Comments 5: Closed-loop navigation safety, recovery, and dynamic-obstacle evaluation

> "The experimental evaluation relies mainly on offline classification metrics, while closed-loop navigation safety and recovery capability are insufficiently validated. Since the framework targets autonomous off-road deployment, additional experiments involving dynamic obstacles, terrain discontinuities, and severe sensor degradation are necessary."

**Response 5:** This is the same scope point raised by reviewer 1 and is already documented honestly in Section 7 (Simulation-Based Evaluation and Limitations). The reviewer is correct that the present manuscript does not yet validate closed-loop navigation safety, dynamic-obstacle reaction, terrain-discontinuity behaviour, or sensor-degradation recovery — and that all of these are necessary before claiming deployment readiness.

The plan documented in Section 7 has two stages, which we will keep and extend in light of reviewer 3's specific request:

1. **Stage 1 (simulation-in-the-loop)**: replay RELLIS-3D LiDAR frames into a physics simulator (Gazebo Classic with a Warthog model), execute the predicted action, and re-render the next LiDAR observation. This is the cheapest way to obtain closed-loop metrics (success rate, intervention rate, deviation from a reference path, time-to-completion) and to inject the three failure conditions reviewer 3 lists: dynamic obstacles (drop a moving obstacle into the simulator), terrain discontinuities (introduce a sudden surface-class change), and severe sensor degradation (mask a LiDAR sector, inject ring-pattern noise, add fog).

2. **Stage 2 (field deployment)**: a real Warthog with the same OS1-64 LiDAR, running the trained model behind a safety driver, on an instrumented off-road course. This is the only stage that establishes deployment readiness; the simulation results are necessary but not sufficient.

We will expand Section 7 to enumerate the specific Stage-1 experiments reviewer 3 requests as concrete future-work items:

- E1: dynamic-obstacle reaction (insert a moving pedestrian/vehicle, measure decision-latency and stop-action triggering)
- E2: terrain-discontinuity transition (replay sequence segments at the boundary between two surface classes, measure action-stream stability across the transition)
- E3: severe sensor degradation (LiDAR sector mask, point noise injection at varying densities, measure traversability-map degradation and downstream action-stream stability)
- E4: recovery capability (simulate a brief sensor outage, measure time-to-recovery of the decision stream after the outage ends)

These experiments are blocked on the Stage-1 simulator integration, which is the next deliverable in the project's roadmap (Section 7).

## 4. Response to Comments on the Quality of English Language

**Point 1:** No specific English-language comments were raised by reviewer 3.

**Response 1:** The manuscript has nonetheless been edited in the current revision to remove AI-writing-style patterns and to vary sentence-length burstiness; the substantive numerical claims are unchanged.

## 5. Additional clarifications

Three notes the reviewer should be aware of:

1. **Empirical-versus-rationale tagging.** Section 6 (Ablation Studies) already labels each ablation as empirical or design-rationale, which is intended to make the strength of evidence behind each architectural claim immediately readable. We will extend this tagging to cover the component-wise decomposition (Response 1), the planned pose-noise injection (Response 4), and the closed-loop sub-experiments E1--E4 (Response 5) once those measurements are available.

2. **Compute budget for the planned ablations.** The four component-removal retrains in Response 1 plus the pose-noise injection in Response 4 plus the Stage-1 simulator integration in Response 5 together require approximately 12--18 hours of GPU time and roughly two engineer-weeks of simulator integration work. We will pursue them in the order: pose-noise (cheap, no retrain), component-removal retrains, then simulator integration. The next revision will report whichever of these are complete at submission time.

3. **Reference \cite{star_inertial_fusion}.** The suggested reference "Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method" will be added to the bibliography. The connection to TerrainFormer is at the deployment-platform layer (IMU fallback for pose during primary-sensor outage), not at the network-architecture layer, and the response above is careful to frame it that way.
