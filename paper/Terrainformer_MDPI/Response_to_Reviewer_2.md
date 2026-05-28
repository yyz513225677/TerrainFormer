# Response to Reviewer 2 Comments

## 1. Summary

Thank you for the constructive review and for pointing out three concrete additions that would make the manuscript easier to position within the literature and easier for readers to evaluate component-by-component. We have implemented all three requests in the revised manuscript: a methodological comparison table that explains the architectural origins of the performance differences between TerrainFormer and prior systems, a technology-tree taxonomy figure that places TerrainFormer in the broader off-road navigation literature, and a consolidated dedicated section titled "Ablation Studies" that gathers the component-level analyses previously scattered across the methodology, results, and discussion sections. The page/section/line pointers below refer to the revised manuscript; all new content is highlighted in the re-submitted files.

## 2. Questions for General Evaluation

| Reviewer's Evaluation | Response and Revisions |
|---|---|
| Does the introduction provide sufficient background and include all relevant references? | Now improved → the new Section 2.5 ("Position of TerrainFormer in the Literature") and its accompanying technology-tree figure (Fig.\ 2) make the relationship between TerrainFormer and four lines of prior work (classical / DL-perception / end-to-end IL / world-model + sequence policy) explicit. |
| Are all the cited references relevant to the research? | Yes — the same Section 2.5 also threads the cited references into the taxonomy so the reader can see which prior work each branch of the tree corresponds to. |
| Is the research design appropriate? | Yes — the methodological comparison tables (Tables 1 and 2) now make the design choices explicit relative to seven representative comparator systems, allowing reviewers to judge each choice in context. |
| Are the methods adequately described? | Yes, and further improved → the new Ablation Studies section (Section 6) consolidates the encoder choice, predictive evaluation, focal-loss recipe, action-chunk size, and goal-direction lookahead into one labelled location with explicit empirical/design-rationale tagging. |
| Are the results clearly presented? | Yes, with the addition that the ablation-style content (PointPillars vs.\ PointNet++ table, predictive evaluation table, focal-loss discussion) is now both indexable from one dedicated section and referenced back from its original analysis context. |
| Are the conclusions supported by the results? | Yes — the Ablation Studies section's summary table (Table 9) explicitly flags which design choices are supported by empirical measurement and which are reported as design rationale, so the strength of evidence behind each claim is readable from one place. |

## 3. Point-by-point response to Comments and Suggestions for Authors

### Comments 1: Methodological Comparison Table

> "Please expand the current evaluation beyond simple performance metrics. Include a methodological comparison table that explicitly outlines the underlying architectural or algorithmic origins of these performance differences which can be referenced by readers."

**Response 1:** Agreed. We have added two methodological comparison tables (Tables 1 and 2, labels `tab:method_arch` and `tab:method_caps`) at the end of Related Work (new Section 2.5 "Position of TerrainFormer in the Literature"). The split into two tables is deliberate so that each table fits inside the MDPI single-column page width: Table 1 reports the architectural backbones (domain, perception, policy) and Table 2 reports the deployment-relevant capabilities (action representation, cross-dataset training, real-time inference). Together they cover six methodological axes across seven representative comparator systems: BADGR \cite{kahn2021}, TartanDrive \cite{triest2023}, Dreamer / DreamerV3 \cite{hafner2020, hafner2023}, MILE \cite{hu2022}, PCWM \cite{yang2023}, the original Decision Transformer \cite{chen2021}, and Wayformer \cite{nayakanti2023}. The System column repeats across the two tables and serves as the linking key so a reader can correlate rows between them.

After the table, three explanatory paragraphs unpack the architectural sources of the performance differences:

1. BADGR and TartanDrive are the closest off-road comparators, both end-to-end. Neither separates terrain representation from action selection, which is what makes our cross-dataset measurement possible at all.
2. MILE and PCWM bring world-model machinery to urban driving with the same BEV-then-transformer pattern. The architectural lineage is acknowledged; the off-road discrete-action vocabulary and cross-dataset protocol are new contributions of TerrainFormer.
3. The original Decision Transformer operates on state vectors, not perception. Pairing it with a BEV world model is the architectural step that turns it into a perception-aware policy.

Inserted text (Section 2.5, end of Related Work):

> "Tables 1 and 2 compare TerrainFormer against representative systems on the architectural and methodological axes that drive their performance differences. Table 1 reports the architectural backbones (domain, perception, policy); Table 2 reports the deployment-relevant capabilities (action representation, cross-dataset training, real-time inference). The point of the comparison is not to claim TerrainFormer is uniformly better; several of the listed systems were designed for urban driving rather than off-road and use different action representations, sensor stacks, or training datasets. What the tables do show is which design choices are shared, which are specific to TerrainFormer, and where each design choice comes from in the literature."

### Comments 2: Technology Tree

> "Provide a technology tree or taxonomy of related work to clearly illustrate where 'this research' fits within the broader literature."

**Response 2:** Agreed. We have added Figure 1 (`fig:tech_tree`) in the same new Section 2.5. The figure is now the first figure of the manuscript because Section 2.5 precedes the architecture overview; the four-branch taxonomy is:

- **Classical / geometric** — \cite{papadakis2013, sock2016}
- **Deep-learning perception** — semantic segmentation \cite{jiang2021, wigness2019} and traversability \cite{wellhausen2019, frey2023, castro2023, frey2024}
- **End-to-end imitation learning** — RGB \cite{bojarski2016, codevilla2018} and LiDAR / multi-modal \cite{kahn2021, triest2023}
- **World-model + sequence policy** — latent dynamics \cite{hafner2020, hafner2023}, BEV world models \cite{hu2022, yang2023}, decision transformers \cite{chen2021, zheng2022, janner2021}, and **TerrainFormer (this work)** at the leaf

TerrainFormer's slot is highlighted in red. The tree distinguishes two kinds of relationships: solid arrows indicate the parent-child taxonomy structure, while dashed red arrows mark the prior components TerrainFormer reuses across branches (BEV world models from urban driving, decision-transformer architectures from offline RL, and self-supervised perception heads from DL-perception). The figure makes both lineage and cross-cutting reuse readable at a glance, which the textual related-work alone did not.

The figure file is the vector PDF `tech_tree.pdf` compiled from `tech_tree.tex` (a standalone TikZ source); it scales cleanly at journal print resolution.

### Comments 3: Ablation Studies

> "The ablation analysis should be strengthened. While related content is currently scattered throughout the text, it would be much clearer if consolidated under a dedicated section titled 'Ablation Study.'"

**Response 3:** Agreed, and addressed by creating a new Section 6 "Ablation Studies" between Results (Section 5) and Discussion (Section 7). The section is intentionally structured to make the strength of evidence behind each ablation explicit (Table 9 tags each row as *empirical* or *design rationale*).

Section structure:

- **6.1 Encoder backbone (PointPillars vs.\ PointNet++)**: *empirical*. References Table 3 from Section 3.1.2 and unpacks the three architectural factors (memory layout / hierarchy depth / output-format alignment) that explain the latency gap.
- **6.2 Predictive evaluation (ground-truth vs.\ predicted observations)**: *empirical*. References Table 8 from Section 5.4; the 0.79\,\% accuracy drop with 98.82\,\% agreement is the load-bearing empirical ablation showing that the decision transformer reads its action signal from the learned latent terrain representation, not from short-term sensor noise.
- **6.3 Focal loss with inverse-frequency class weighting**: *partially empirical*. We ran one CE-only Phase 2 training run during early development and observed near-zero recall on every minority class with accuracy in the 60--65\,\% range. Switching to focal loss with automatic inverse-frequency weights ($\gamma{=}2.0$, label smoothing $0.1$) is what lifts every per-class F1 above 0.65.
- **6.4 Action-chunk size $K$**: *design rationale*. We use $K{=}5$ matching the original action-chunking paper~\cite{zhao2023act}. The two failure modes at the extremes ($K$ too small loses look-ahead, $K$ too large amplifies prediction error) are described.
- **6.5 Goal-direction lookahead $k$**: *design rationale*. We use $k{=}5$ to match the chunk-prediction horizon. The two failure modes ($k{=}1$ becomes pose noise, $k{=}20$ becomes uncorrelated with the current step) are described.
- **6.6 Summary table** (`tab:ablation_summary`): a single overview table tagging each ablation as empirical or design-rationale and noting the measured effect.

Honest scope statement included at the top of Section 6: "We did not run a full multi-seed retrain for every variant because of the compute cost (Section~\ref{sec:limitations}), so we are explicit about which entries below are empirical and which are qualitative." This makes it clear which claims are backed by measurement and which by argued design rationale, rather than presenting the section as if every ablation had been exhaustively measured.

Inserted text (Section 6, opening):

> "The design space of TerrainFormer contains five choices we evaluated explicitly: the LiDAR encoder backbone, the predictive-evaluation protocol (substituting predicted future frames for ground-truth observations), the focal-loss training recipe, the action-chunk size $K$, and the goal-direction lookahead $k$. The first two are empirically measured; the latter three are reported as design-choice ablations supported by the design's failure modes when each component is removed."

## 4. Response to Comments on the Quality of English Language

**Point 1:** No specific comments on English-language quality were raised by this reviewer.

**Response 1:** We have nevertheless taken the opportunity of this revision pass to humanise the prose of the abstract, introduction, contributions list, and conclusion, removing AI-writing-style patterns (significance inflation, formulaic "principal contribution" framing, uniform sentence length, generic positive conclusions). The numerical claims are unchanged. The pass is described in the commit history of the manuscript source.

## 5. Additional clarifications

Three items related to the new content that the reviewer should be aware of:

1. **Multi-seed retrain is deferred.** Sections 6.3, 6.4, and 6.5 are tagged as design-rationale rather than empirical because running the full +/-focal-loss, +/-action-chunk, +/-goal-direction Phase 2 variants requires approximately 8--12 hours of additional GPU time (each variant is one full Phase 2 retrain). This is documented honestly in Section 6 and in the Limitations subsection. We are happy to run any specific ablation variant the reviewer prioritises in a subsequent revision.

2. **The technology-tree figure is a vector PDF.** It compiles from a TikZ standalone source (`tech_tree.tex`) included in the submission archive, so it can be regenerated cleanly if the journal requests typographic changes.

3. **The methodology comparison table covers seven representative systems.** It is not exhaustive; the choice prioritises systems that share at least one architectural component with TerrainFormer (BEV input, world model, transformer policy, off-road domain) so that the comparison illuminates design choices rather than enumerating the entire off-road navigation literature.
