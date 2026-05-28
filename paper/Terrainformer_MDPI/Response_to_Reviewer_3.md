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

We will address this in the next revision via empirical decomposition: run four additional Phase 2 retrains, each with one component removed: (i) world model off (decision transformer reads raw BEV features), (ii) action chunking off ($K{=}1$, no TemporalEnsemble), (iii) TemporalEnsemble off ($K{=}5$ but argmax-per-frame instead of weighted average), and (iv) cross-dataset protocol off (Phase 1 also trained on RELLIS-3D). Each retrain is approximately 2 hours on a Quadro RTX 8000; the total compute is approximately 8 hours. Results will form a new subsection 6.3a "Component-wise decomposition" within the existing Ablation Studies section, with the per-component accuracy delta reported against the full-model baseline so that the dominant architectural contributors to cross-dataset generalisation are explicit. The planned experiments are clearly marked in Section 6 of the revised manuscript.

### Comments 2: Sensor fusion under degraded perception (inertial / star-sensor reference)

> "The navigation framework would benefit from stronger discussion of integrated inertial sensing robustness under degraded perception conditions. The authors should add some discussions about navigations, such as 'Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method,' particularly regarding sensor fusion reliability, alignment stability, and fault-tolerant navigation initialization in complex environments."

**Response 2:** We agree that integrated inertial-sensing robustness under degraded perception is an important question for any deployable off-road navigation system. However, it falls outside the present manuscript's scope: TerrainFormer is a learning-architecture contribution evaluated on offline LiDAR data with ground-truth poses, not a deployment-platform sensor-fusion study. Reliably addressing inertial / star-sensor fusion reliability, alignment stability, and fault-tolerant initialisation would require its own experimental apparatus (a vehicle with a calibrated IMU stack, controlled sensor-degradation conditions, etc.) that is not part of the present study. We therefore add this topic to the explicit Future Works list in the renamed Section 8 ("Conclusion and Future Works") of the revised manuscript, citing the suggested reference \cite{star_inertial_fusion} as the starting point for that future work.

### Comments 3: Cooperative motion stability and multi-step control under rapid terrain changes

> "Although the decision transformer demonstrates promising trajectory consistency, the manuscript does not sufficiently analyze cooperative motion stability and dynamic interaction during complex maneuvers. The authors should discuss connections with temporal smoothness constraints, anticipatory trajectory adaptation, and multi-step control stability under rapidly changing terrain conditions."

**Response 3:** We agree the cooperative-motion-stability and multi-step control analysis would strengthen the deployment case for TerrainFormer. However, the metrics the reviewer asks about — cross-track error growth rate, attitude oscillation amplitude, action-stream cross-correlation under terrain-class boundary events — require closed-loop evaluation, which is beyond the scope of the present open-loop study. The current manuscript's per-frame imitation metrics cannot exhibit divergence from the recorded trajectory because the vehicle's motion is fixed by the original recording. We therefore add cooperative motion stability, anticipatory trajectory adaptation, and multi-step control stability under terrain-class boundary events to the Future Works list in Section 8.

### Comments 4: Sensor precision and navigation drift impact on decision quality

> "The manuscript emphasizes real-time off-road navigation accuracy, yet the influence of sensor precision and navigation drift on downstream decision quality is not rigorously analyzed. The authors need to add some analyses about this part, particularly regarding high-precision attitude stabilization and its implications for robust autonomous navigation performance."

**Response 4:** We agree that the impact of sensor precision and accumulated pose drift on downstream decision quality is a relevant question. However, the present study uses ground-truth poses from the RELLIS-3D SLAM solution and does not run controlled noise-injection experiments; a rigorous analysis would require a separate experimental design (controlled pose-noise injection, IMU-error modelling, high-precision attitude-stabilisation comparison) that is outside the present manuscript's scope. We add this analysis — including the controlled pose-noise injection experiment and its connection to high-precision attitude stabilisation — to the Future Works list in Section 8.

### Comments 5: Closed-loop navigation safety, recovery, and dynamic-obstacle evaluation

> "The experimental evaluation relies mainly on offline classification metrics, while closed-loop navigation safety and recovery capability are insufficiently validated. Since the framework targets autonomous off-road deployment, additional experiments involving dynamic obstacles, terrain discontinuities, and severe sensor degradation are necessary."

**Response 5:** We agree, and we note that the present manuscript is explicit about this scope limit (Section 7's "Simulation-Based Evaluation and Limitations" subsection states that the current evaluation is open-loop). Reliably validating closed-loop navigation safety, recovery capability, dynamic-obstacle reaction, terrain-discontinuity behaviour, and sensor-degradation recovery requires a closed-loop apparatus (simulation-in-the-loop on Gazebo / Isaac Sim, or field deployment on a physical Warthog) that is outside the scope of the present learning-architecture paper. We therefore add the four specific closed-loop experiments (dynamic obstacles, terrain discontinuities, severe sensor degradation, recovery capability) to the Future Works list in Section 8.

## 4. Response to Comments on the Quality of English Language

**Point 1:** No specific English-language comments were raised by reviewer 3.

**Response 1:** The manuscript has nonetheless been edited in the current revision to remove AI-writing-style patterns and to vary sentence-length burstiness; the substantive numerical claims are unchanged.

## 5. Additional clarifications

Three notes the reviewer should be aware of:

1. **Empirical-versus-rationale tagging.** Section 6 (Ablation Studies) already labels each ablation as empirical or design-rationale, which is intended to make the strength of evidence behind each architectural claim immediately readable. We will extend this tagging to cover the component-wise decomposition (Response 1), the planned pose-noise injection (Response 4), and the closed-loop sub-experiments E1--E4 (Response 5) once those measurements are available.

2. **Compute budget for the planned ablations.** The four component-removal retrains in Response 1 plus the pose-noise injection in Response 4 plus the Stage-1 simulator integration in Response 5 together require approximately 12--18 hours of GPU time and roughly two engineer-weeks of simulator integration work. We will pursue them in the order: pose-noise (cheap, no retrain), component-removal retrains, then simulator integration. The next revision will report whichever of these are complete at submission time.

3. **Reference \cite{star_inertial_fusion}.** The suggested reference "Enhancing attitude availability in star-depleted cases: an inertial/star sensor fusion method" will be added to the bibliography. The connection to TerrainFormer is at the deployment-platform layer (IMU fallback for pose during primary-sensor outage), not at the network-architecture layer, and the response above is careful to frame it that way.
