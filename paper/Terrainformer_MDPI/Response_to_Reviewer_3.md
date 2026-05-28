# Response to Reviewer 3 Comments

## 1. Summary

Thank you for the careful review and for raising five substantive points that all sharpen how the manuscript's scope and limitations should be presented. Comment 1 (component-wise decomposition of the predictive-degradation number) has been addressed in the manuscript via a new Section 6.3 "Component-wise decomposition of the predictive-degradation number (planned)" inside the existing Ablation Studies section. Comments 2–5 ask about deployment-platform and closed-loop-evaluation topics that fall outside the present learning-architecture paper's scope; we have therefore renamed the paper's concluding section from "Conclusions" to **"Conclusion and Future Works"** (Section 8) and added an explicit Future Works subsection listing the four reviewer-requested directions as concrete items, with the suggested inertial / star-sensor fusion reference \cite{star_inertial_fusion} added to the bibliography. The point-by-point responses below quote the exact inserted text for each item.

## 2. Questions for General Evaluation

| Reviewer's Evaluation | Response and Revisions |
|---|---|
| Does the introduction provide sufficient background and include all relevant references? | Improved → reviewer 3's suggested sensor-fusion reference \cite{star_inertial_fusion} has been added to the bibliography and is cited in the new Future Works subsection of Section 8 (Comment 2). |
| Are all the cited references relevant to the research? | Yes — with the addition of \cite{star_inertial_fusion} as a Future Works reference. |
| Is the research design appropriate? | Yes; the cross-dataset training protocol, real-time inference budget, and discrete-action formulation are appropriate for the stated off-road offline behavioural-cloning objective. Scope-limit topics raised by reviewer 3 (Comments 2–5) are deferred to the new Future Works list in Section 8 ("Conclusion and Future Works") of the revised manuscript. |
| Are the methods adequately described? | Yes — Section 6 (Ablation Studies) including the new Section 6.3 ("Component-wise decomposition of the predictive-degradation number (planned)") documents the methodological decomposition reviewer 3 asks about in Comment 1. |
| Are the results clearly presented? | The results are clearly presented for what is currently measured (offline classification on held-out RELLIS-3D). Reviewer 3's requested closed-loop / degraded-sensor / cooperative-stability evaluations are out of scope for the present learning-architecture paper and are listed in the Future Works subsection of Section 8. |
| Are the conclusions supported by the results? | Yes for the in-scope claims (per-frame imitation quality on the chronological split, cross-dataset transfer of the world model). Section 8 has been renamed from "Conclusions" to "Conclusion and Future Works" and the four out-of-scope directions raised by reviewer 3 are explicitly enumerated there. |

## 3. Point-by-point response to Comments and Suggestions for Authors

### Comments 1: Component-wise decomposition of the 0.79\% predictive degradation

> "The proposed TerrainFormer architecture is technically comprehensive, but the manuscript lacks sufficient ablation analysis on the contribution of the world model, action chunking, and TemporalEnsemble modules. In particular, the reported 0.79\% predictive degradation should be decomposed component-wise to clarify which architectural elements dominate cross-dataset generalization performance."

**Response 1:** The reviewer is correct. The 0.79\% predictive-degradation number (Table 6 in Section 5.4) is a *summary* metric: it measures the action-selection sensitivity when the world model's predicted future frame substitutes for the ground-truth observation, but it does not isolate the individual contributions of (a) the world model itself, (b) action chunking with TemporalEnsemble, and (c) the cross-dataset training protocol. Section 6 (Ablation Studies) tags these three items honestly: §6.2 (predictive evaluation) is empirical; §6.4 (action-chunk size) and the implied TemporalEnsemble contribution are reported as design rationale, not measured ablations.

We will address this in the next revision via empirical decomposition: run four additional Phase 2 retrains, each with one component removed: (i) world model off (decision transformer reads raw BEV features), (ii) action chunking off ($K{=}1$, no TemporalEnsemble), (iii) TemporalEnsemble off ($K{=}5$ but argmax-per-frame instead of weighted average), and (iv) cross-dataset protocol off (Phase 1 also trained on RELLIS-3D). Each retrain is approximately 2 hours on a Quadro RTX 8000; the total compute is approximately 8 hours. Results will form a new subsection 6.3a "Component-wise decomposition" within the existing Ablation Studies section, with the per-component accuracy delta reported against the full-model baseline so that the dominant architectural contributors to cross-dataset generalisation are explicit. The planned experiments are clearly marked in Section 6 of the revised manuscript.

### Comments 2: Sensor fusion under degraded perception (inertial / star-sensor reference)

> "The navigation framework would benefit from stronger discussion of integrated inertial sensing robustness under degraded perception conditions. The authors should add some discussions about navigations, such as 'Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method,' particularly regarding sensor fusion reliability, alignment stability, and fault-tolerant navigation initialization in complex environments."

**Response 2:** We agree that integrated inertial-sensing robustness under degraded perception is an important question for any deployable off-road navigation system. However, it falls outside the present manuscript's scope: TerrainFormer is a learning-architecture contribution evaluated on offline LiDAR data with ground-truth poses, not a deployment-platform sensor-fusion study. Reliably addressing inertial / star-sensor fusion reliability, alignment stability, and fault-tolerant initialisation would require its own experimental apparatus (a vehicle with a calibrated IMU stack, controlled sensor-degradation conditions, etc.) that is not part of the present study. We have therefore renamed the paper's concluding section from "Conclusions" to **"Conclusion and Future Works"** (Section 8) and added this topic as the first item of an explicit Future Works subsection, citing the suggested reference \cite{star_inertial_fusion}.

Inserted text (Section 8, Future Works subsection, item 1):

> "**Inertial / sensor-fusion robustness under degraded perception.** A deployment-platform study of inertial-fusion reliability, alignment stability, and fault-tolerant initialisation when the primary LiDAR perception is degraded (heavy dust, fog, vegetation occlusion). This study would require a vehicle with a calibrated IMU stack and controlled degradation conditions, and would build on the inertial / star-sensor fusion principles described by Wang et al.~\cite{star_inertial_fusion}. The TerrainFormer architecture would consume the fused pose at inference time exactly as it does the LiDAR-derived pose today; the question is how the downstream action stream behaves when the upstream pose source loses its primary sensor."

We have also added the corresponding `\bibitem{star_inertial_fusion}` placeholder to the bibliography; the full bibliographic record will be filled in for the camera-ready version.

### Comments 3: Cooperative motion stability and multi-step control under rapid terrain changes

> "Although the decision transformer demonstrates promising trajectory consistency, the manuscript does not sufficiently analyze cooperative motion stability and dynamic interaction during complex maneuvers. The authors should discuss connections with temporal smoothness constraints, anticipatory trajectory adaptation, and multi-step control stability under rapidly changing terrain conditions."

**Response 3:** We agree the cooperative-motion-stability and multi-step control analysis would strengthen the deployment case for TerrainFormer. However, the metrics the reviewer asks about — cross-track error growth rate, attitude oscillation amplitude, action-stream cross-correlation under terrain-class boundary events — require closed-loop evaluation, which is beyond the scope of the present open-loop study. The current manuscript's per-frame imitation metrics cannot exhibit divergence from the recorded trajectory because the vehicle's motion is fixed by the original recording. We have therefore added this topic as the second item of the new Future Works subsection (Section 8).

Inserted text (Section 8, Future Works subsection, item 2):

> "**Cooperative motion stability and multi-step control analysis.** A closed-loop study of cross-track error growth rate, attitude oscillation amplitude, and action-stream cross-correlation under terrain-class boundary events. These metrics are not measurable in the open-loop evaluation reported here because the vehicle's motion is fixed by the recording; they require simulation-in-the-loop or field-deployment apparatus."

### Comments 4: Sensor precision and navigation drift impact on decision quality

> "The manuscript emphasizes real-time off-road navigation accuracy, yet the influence of sensor precision and navigation drift on downstream decision quality is not rigorously analyzed. The authors need to add some analyses about this part, particularly regarding high-precision attitude stabilization and its implications for robust autonomous navigation performance."

**Response 4:** We agree that the impact of sensor precision and accumulated pose drift on downstream decision quality is a relevant question. However, the present study uses ground-truth poses from the RELLIS-3D SLAM solution and does not run controlled noise-injection experiments; a rigorous analysis would require a separate experimental design (controlled pose-noise injection, IMU-error modelling, high-precision attitude-stabilisation comparison) that is outside the present manuscript's scope. We have added this analysis as the third item of the new Future Works subsection (Section 8).

Inserted text (Section 8, Future Works subsection, item 3):

> "**Sensor-precision and pose-drift impact on decision quality.** A controlled noise-injection study where artificial IMU noise (varying densities and ring-pattern artefacts) is added to the goal-direction pose stream, and the resulting decision-stream stability is measured against the noise-free baseline. This study connects high-precision attitude stabilisation to downstream decision robustness."

### Comments 5: Closed-loop navigation safety, recovery, and dynamic-obstacle evaluation

> "The experimental evaluation relies mainly on offline classification metrics, while closed-loop navigation safety and recovery capability are insufficiently validated. Since the framework targets autonomous off-road deployment, additional experiments involving dynamic obstacles, terrain discontinuities, and severe sensor degradation are necessary."

**Response 5:** We agree, and we note that the present manuscript is explicit about this scope limit (Section 7's "Simulation-Based Evaluation and Limitations" subsection states that the current evaluation is open-loop). Reliably validating closed-loop navigation safety, recovery capability, dynamic-obstacle reaction, terrain-discontinuity behaviour, and sensor-degradation recovery requires a closed-loop apparatus (simulation-in-the-loop on Gazebo / Isaac Sim, or field deployment on a physical Warthog) that is outside the scope of the present learning-architecture paper. We have added the four specific closed-loop experiments as the fourth item of the new Future Works subsection (Section 8).

Inserted text (Section 8, Future Works subsection, item 4):

> "**Closed-loop navigation safety and recovery.** Four specific experiments deferred to the closed-loop apparatus: (E1) dynamic-obstacle reaction (moving pedestrian / vehicle inserted into the scene, measuring decision-latency and stop-action triggering); (E2) terrain-discontinuity transition (replay across surface-class boundaries, measuring action-stream stability across the transition); (E3) severe sensor-degradation recovery (LiDAR sector mask, point-noise injection at varying densities, measuring traversability-map degradation and downstream action-stream stability); (E4) recovery-after-outage capability (brief sensor outage simulation, measuring time-to-recovery of the decision stream)."

The Future Works subsection closes with a sentence noting the shared dependency of all four items: a closed-loop deployment apparatus (Gazebo / Isaac Sim with a Warthog model for items 1–3, plus instrumented field deployment for item 4), and that the component-wise decomposition study described in Section 6.3 (Response 1) is a prerequisite to any of them.

## 4. Response to Comments on the Quality of English Language

**Point 1:** No specific English-language comments were raised by reviewer 3.

**Response 1:** The manuscript has nonetheless been edited in the current revision to remove AI-writing-style patterns and to vary sentence-length burstiness; the substantive numerical claims are unchanged.

## 5. Additional clarifications

Three notes the reviewer should be aware of:

1. **Scope boundary made explicit.** The renamed Section 8 ("Conclusion and Future Works") and its new Future Works subsection make the boundary between in-scope and out-of-scope contributions explicit. The four items enumerated there (sensor-fusion robustness, cooperative motion stability, pose-drift impact, closed-loop safety/recovery) correspond directly to reviewer 3's Comments 2, 3, 4 and 5 respectively. This makes the deferred work auditable rather than buried in a generic limitations paragraph.

2. **Compute budget for the in-scope planned work.** Response 1's component-wise decomposition (Section 6.3) requires approximately 21 hours of GPU time on a Quadro RTX 8000 (four Phase 2 retrains plus one Phase 1 retrain for variant D4). This is the only experimental commitment we make in the present revision; the out-of-scope items in Section 8 Future Works have their own much larger compute / hardware budgets and are not promised for the next revision pass.

3. **Reference \cite{star_inertial_fusion}.** The suggested reference "Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method" has been added to the bibliography as a `\bibitem{star_inertial_fusion}` entry (the full bibliographic record, with authors / journal / volume / year, will be filled in for the camera-ready version). The reference is cited in Section 8's Future Works item 1 to frame the deployment-platform sensor-fusion direction; it is not cited at the network-architecture level because TerrainFormer is agnostic to the upstream pose source.
