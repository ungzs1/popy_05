# Behavioral effects of the chemogenetic activation of the LC/NA→MCC pathway (DCZ), monkey YU

**Data and statistics.** 20 DCZ and 23 sham (saline) sessions, interleaved over the same weeks (Sep 2023 – Feb 2024), with 23 no-injection sessions for reference. Task parameters are identical in all sessions (lever hold 1.5 s, target hold 450 ms, session ends after 350 rewards; checked in the csv headers). One value per session, completed trials only. DCZ vs sham: two-sided Mann-Whitney U, rank-biserial r; robustness check by OLS adjusted for session date and weekday; Benjamini-Hochberg FDR over all 30 DCZ vs sham tests (q). All numbers and figures below come from sections (6) and (7) of `changing_strategy_yu.ipynb` (figure files: `figs/yu_dcz_*.svg`, regenerated when the notebook is run).

**Figures.** Colors: none = grey, sham = blue, DCZ = green. In the box plots each dot is a session; panel titles give the DCZ vs sham Mann-Whitney p (uncorrected) and the rank-biserial r. Line plots show the mean ± SEM over sessions.

## Main effects of DCZ

1. **Earlier release of the lever, not faster movement.** RT (lever validation → release) drops from 346 to 247 ms (−29 %), and the probability of releasing the lever in the last 500 ms before the 1.5 s validation doubles (0.11 → 0.24). Movement time and accidental early slips are unchanged. These are the only effects that survive the FDR correction (q < 0.001). DCZ seems to shift the timing of the lever release earlier, as in anticipation or urgency, rather than speeding up the movement or making the animal more impulsive in general.
2. **Less disengagement after unrewarded trials** (nominal p ≈ 0.01, q ≈ 0.06). After an unrewarded trial, the slowing before the next lever touch is smaller and long pauses (> 5 s) are about half as frequent. After rewarded trials nothing changes.
3. **More persistent, less drifting choice behavior, with unchanged efficiency** (nominal p < 0.05, q ≈ 0.06-0.13). Switching does not increase over the session under DCZ. The foraging model has a lower switch threshold and a lower inverse temperature, which means less switching when the current target has a low value. Outcomes 4-6 trials back weigh less on switching. The reward rate, P(best target) and the speed of finding the best target after a block change are unchanged.
4. **Energized but value-blind responses** (exploratory, section (7)). DCZ accelerates all lever releases, including those that react to the validation sound. The speed of the release no longer tracks the value of the current target or the upcoming choice as it does in sham sessions (q < 0.001). The effect is present in every DCZ session, without tolerance over the months.

## Hypotheses and results

**A. Energization of action**

- **H1. DCZ shortens decision RT.** Increasing NA makes monkeys execute trials faster and more consistently (Kaduk et al., 2023, as cited in the manuscript). Lowering NA with clonidine reduces energization (Jahn et al., 2018; Borderies et al., 2020), and LC neurons are sensitive to effort (Varazzani et al., 2015).
  Result: median RT 346 ms (sham) → 247 ms (DCZ), none 320 ms; p = 2×10⁻⁶, q < 0.001, r = −0.85, adjusted OLS p = 10⁻⁹. RT is also shorter in every quarter of the session. **Supported.**
- **H2. Motor execution is not affected** (cognitive vs motor dissociation proposed in the manuscript discussion).
  Result: movement time (release → first target touch) 227 vs 227 ms; p = 0.80, r = −0.05. **Supported:** the speeding is specific to the decision / release.

![Figure 1 (H1, H2).](figs/yu_dcz_A_rt_mt.svg)

*Figure 1 (H1, H2). Session median of log RT (left) and of log movement time (middle). Right: session median RT across the recording period.*

- **H3. The faster RT is part of an earlier lever release overall (temporal anticipation, urgency).** Pupil-linked (putatively noradrenergic) gain modulation produces time-dependent urgency (Murphy et al., 2016). The opposite prediction comes from rodents, where raising NA reduces premature responses: atomoxetine (Robinson et al., 2008) and optogenetic LC stimulation (Bari et al., 2020). In mice, response inhibition depends on the LC→ventrolateral OFC projection, not on LC→dmPFC (Bari et al., 2020).
  Result: P(release in the last 500 ms of the hold | held until then) 0.11 → 0.24 (none 0.18); p = 8×10⁻⁶, q < 0.001, r = +0.80. Releases in the first 250 ms of the hold (slips) are unchanged (0.28 vs 0.26, p = 0.62). Early lever releases per trial rise from 0.66 to 0.95 (session medians).
  **Supported.** The extra lever presses under DCZ are re-presses after anticipatory releases, not extra trial initiations, so they should not be read as more engagement. RT is measured on validated holds only, so moving the earliest releases before validation makes the RT decrease a conservative estimate.

![Figure 2 (H3).](figs/yu_dcz_A_lever_hold.svg)

*Figure 2 (H3). Left: hazard of releasing the lever before validation, by time holding the lever (log scale; validation at 1.5 s). Middle: probability of releasing in the last 500 ms before validation. Right: probability of releasing in the first 250 ms of the hold (slips, control).*

**B. Engagement and time-on-task**

- **H4. DCZ increases task engagement:** fewer pauses, faster trial initiation, higher work rate. MCC→dlPFC activation increases engagement (Gandaux et al., 2025); the adaptive gain theory links LC mode to task engagement (Aston-Jones & Cohen, 2005).
  Result:
  - Pauses > 5 s before touching the lever: 0.9 % → 0.4 % of trials (none 2.2 %); p = 0.017, q = 0.064, adjusted p = 0.043.
  - Median initiation latency: unchanged (p = 0.84).
  - Completed trials per minute: 9.1 vs 9.3 (p = 0.95).

  **Partially supported** (fewer pauses only). The work rate is constrained by the task: the session ends after a fixed number of rewards, and DCZ trials include extra lever re-holds.

![Figure 3 (H4).](figs/yu_dcz_B_engagement.svg)

*Figure 3 (H4). Probability of a pause > 5 s before starting a trial, median log initiation latency (lever available → first lever touch), completed trials per minute.*

- **H5. DCZ reduces the disengagement caused by negative outcomes.** MCC processes negative feedback to drive adaptation (Quilodran et al., 2008), and NA mobilizes resources to face challenges (Jahn et al., 2018; Borderies et al., 2020).
  Result:
  - Slowing of trial initiation after an unrewarded trial: 1.68 → 1.49 ln units (p = 0.012, q = 0.062, adjusted p = 0.029).
  - Pauses after an unrewarded trial: 1.9 % → 1.1 % (p = 0.012, q = 0.062).
  - After rewarded trials, neither initiation latency (p = 0.60) nor pauses (p = 0.59) change.
  - Initiation after an unrewarded trial is shorter both when the monkey then stays (655 → 590 ms, p = 0.023) and when it switches (630 → 540 ms, p = 0.19). This points to a general re-engagement after losses, not a faster deliberation before switching.

  **Supported (nominally).**

![Figure 4 (H5).](figs/yu_dcz_B_post_loss.svg)

*Figure 4 (H5). Slowing of trial initiation after an unrewarded trial (Δ ln ms relative to after a rewarded trial), probability of a pause > 5 s after unrewarded and after rewarded trials, median log initiation latency after rewarded trials (controls).*

- **H6. DCZ reduces time-on-task effects** (manuscript discussion). Disengagement over time on task goes with decreasing pupil-linked arousal (Hopstaken et al., 2015).
  Result:
  - Slope of P(switch) over the session: +0.069 (sham) vs −0.026 (DCZ); p = 0.012, q = 0.062, adjusted p = 0.019.
  - Slope of P(best target): −0.14 vs −0.01, p = 0.26 (n.s.).

  **Partially supported:** switching stays stable over the session under DCZ.
- **H7. Time course consistent with DCZ pharmacology.** DCZ acts within minutes and its effect lasts several hours (Nagai et al., 2020).
  Result:
  - The DCZ − sham RT difference is present from the first quarter of the session (−0.41 ln units) and smaller in the last quarter (−0.25, still p = 0.007). Slope of RT over the session: p = 0.021, q = 0.071.
  - The early-lever-release excess is largest in the first quarter (+0.48 vs +0.16 per trial in the last quarter); slope p = 0.047.

  **Consistent:** rapid onset, and a decline within the ~1 h session. The interval between injection and session start is not in the data, so this could not be checked further.

![Figure 5 (H6, H7).](figs/yu_dcz_B_time_course.svg)

*Figure 5 (H6, H7). Median log RT, early lever releases per trial, P(switch) and P(best target) per quarter of the session.*

![Figure 6 (H6, H7).](figs/yu_dcz_B_slopes.svg)

*Figure 6 (H6, H7). Slope over the session (per-session linear fit against the fraction of the session completed) of P(switch), P(best target), log RT and early lever releases.*

**C. Learning, exploration and persistence**

- **H8. Faster convergence to the best target after a block change** (manuscript discussion). The network reset (Bouret & Sara, 2005) and unexpected uncertainty (Yu & Dayan, 2005) accounts predict faster adaptation to unsignaled changes.
  Result:
  - P(best target) in trials 1-15 of a block: 0.72 vs 0.70 (p = 0.87).
  - P(switch) in trials 1-15: 0.17 → 0.14 (p = 0.082, q = 0.19).
  - Session P(best target) and reward rate: unchanged (0.765 vs 0.765, 0.631 vs 0.625).

  **Not supported** for the speed of convergence and overall efficiency. Reduced early-block switching is a trend.

![Figure 7 (H8).](figs/yu_dcz_C_block_curves.svg)

*Figure 7 (H8). P(best target) and P(switch) by trial in block, after a block change (shaded: trials 1-15).*

![Figure 8 (H8).](figs/yu_dcz_C_block_measures.svg)

*Figure 8 (H8). P(best target) and P(switch) in trials 1-15 of a block; P(best target) and reward rate over the session.*

- **H9. More weight on recent outcomes.** The manuscript's logistic model suggests this, and pupil-linked arousal regulates the learning rate (Nassar et al., 2012). Method: switch regression on the last 10 outcomes, as in Ungvárszki et al. (2025) and Lau & Glimcher (2005).
  Result:
  - Weight of the last outcome: unchanged (p = 0.34).
  - Summed weight of outcomes 4-6 trials back: −0.83 → −0.26 (p = 0.042, q = 0.13; lag 4 alone p = 0.02 uncorrected).
  - Foraging-model learning rate α: 0.44 → 0.50 (p = 0.078).

  **Partially supported:** older outcomes are forgotten faster (shorter memory), rather than the last outcome being weighted more.

![Figure 9 (H9).](figs/yu_dcz_C_history_weights.svg)

*Figure 9 (H9). Left: weights of the outcomes of the last 10 trials on P(switch), logistic regression per session (* DCZ vs sham p < 0.05, uncorrected). Middle: weight of the last outcome. Right: summed weight of the outcomes 4-6 trials back.*

- **H10. DCZ changes decision noise / random exploration.** Predictions point both ways:
  - Increase: LC input to ACC switches rats into stochastic choice (Tervo et al., 2014); a high tonic LC mode promotes exploration (Aston-Jones & Cohen, 2005; Jepma & Nieuwenhuis, 2011).
  - Decrease: clonidine decreases choice variability (Jahn et al., 2018); atomoxetine reduces random exploration (Warren et al., 2017); propranolol reduces value-free random exploration (Dubois et al., 2021).

  Result:
  - Value-free switching (P(switch | rewarded)): 1.2 % vs 1.4 % (p = 0.37).
  - Foraging inverse temperature β: 8.0 → 6.8 (p = 0.008, q = 0.062, adjusted p = 0.009).
  - RL inverse temperature: 3.2 → 2.9 (p = 0.14).

  **Not supported for random exploration.** The lower β is better understood together with H11.
- **H11. DCZ changes persistence (switch threshold).** Adaptive gain theory predicts more disengagement from the current option; the manuscript discussion predicts reduced shifting.
  Result:
  - Foraging switch threshold τ: 0.18 → 0.12 (p = 0.015, q = 0.064, adjusted p = 0.005).
  - With the lower β, the fitted switching function reaches a lower maximal P(switch) when the current target's value is low, i.e. more persistence after repeated losses. P(switch | 4+ losses in a row): 0.72 → 0.65 (p = 0.125).
  - RL stickiness κ: 2.4 → 2.7 (p = 0.12).
  - β and τ are only moderately correlated across sessions (Spearman ρ = 0.44).

  **Partially supported, in the direction of more persistence**, against the adaptive gain prediction.

![Figure 10 (H9-H11).](figs/yu_dcz_C_models.svg)

*Figure 10 (H9-H11). Top: P(switch) after 0 to 4+ unrewarded trials in a row; switching function of the foraging model with the median fitted parameters per condition; learning rate α; P(switch | rewarded). Bottom: log inverse temperature β of the foraging and of the inferential RL model; switch threshold τ; P(switch | 4+ unrewarded trials in a row).*

- **H12. Directed exploration is unchanged.** Atomoxetine affected random but not directed exploration (Warren et al., 2017; task logic: Wilson et al., 2014).
  Result: when switching, P(going to the less recently chosen target) is 0.40 (sham) vs 0.44 (DCZ), p = 0.095; below 0.5 in all conditions. **Supported (no significant change, trend).**

![Figure 11 (H12).](figs/yu_dcz_C_directed_exploration.svg)

*Figure 11 (H12). Probability of choosing the less recently chosen of the two other targets when switching (dashed line: 0.5, no preference).*

## Comparison with the claims of the manuscript discussion

| Claim | This analysis |
|---|---|
| Faster RT without a change in movement time | Reproduced (H1, H2) |
| More lever presses | Reproduced, but explained by anticipatory hold failures (H3) |
| Reduced time-on-task effects | Partly reproduced: switching does not drift over the session (H6) |
| Reduced shifting early in blocks | Trend only (H8, p = 0.08) |
| Faster convergence to the best target | Not reproduced (H8) |
| More weight on recent outcomes | Partly reproduced: older outcomes weigh less, the last outcome does not weigh more (H9) |

## Original hypotheses (exploratory)

These hypotheses are not taken from the manuscript or its discussion. I formulated them after the results above, so they are exploratory. The tests are the same as above (DCZ vs sham sessions), with a separate FDR correction over the 15 tests of this part (section (7) of the notebook).

- **O1. DCZ accelerates the whole release process, rather than adding anticipatory guesses.** The shorter RT could come from more releases that anticipate the validation (guesses made without waiting for its sound), or from faster releases overall, including those that react to the sound.
  Result:
  - Every decile of the RT distribution is shorter under DCZ, by 46 ms at the 10th percentile up to 135 ms at the 90th (p ≤ 2×10⁻⁵ for each decile).
  - A two-component mixture of log RT per session separates fast and slow releases. The share of fast releases is unchanged (0.28 vs 0.26, p = 0.35), but both kinds are earlier: fast releases 152 → 70 ms (p = 10⁻⁴), slow releases 384 → 294 ms (p = 4×10⁻⁶).

  **Supported:** the whole release process is accelerated, including the releases that react to the validation sound. DCZ does not simply add guesses.

![Figure 12 (O1).](figs/yu_dcz_O1_rt_distribution.svg)

*Figure 12 (O1). Left: deciles of the RT distribution. Right: two-component Gaussian mixture of log RT per session: share of the fast releases, mean RT of the fast and of the slow releases.*

- **O2. The extra premature lever releases do not come from a failure of error correction.** The MCC monitors performance, so activating its noradrenergic input could impair the correction after a timing error. If so, premature releases would repeat within a trial.
  Result: the probability of a premature release on the first attempt of a trial rises from 0.27 to 0.40 (p = 5×10⁻⁶, q < 0.001). After a premature release it is 0.59 in both conditions (p = 0.47).

  **Supported:** the correction after a premature release is intact. DCZ raises premature releases only when the monkey has not just made one.

![Figure 13 (O2).](figs/yu_dcz_O2_timing_errors.svg)

*Figure 13 (O2). Probability of a premature lever release on the first attempt of a trial, and after a premature release.*

- **O3. DCZ advances the commitment to the choice.** While holding the lever, the monkey sometimes touches a target (codes 111-113 give which one), which reveals the target it is preparing to choose. If DCZ advances the decision, these touches should be more frequent, and on switch trials they should more often point to the new target.
  Result:
  - Trials with a target touch while holding the lever: 34 % → 41 % (p = 8×10⁻⁴, q = 0.002).
  - The touched target is the chosen one in about 80 % of the trials in both conditions (p = 0.71).
  - On switch trials, the touched target is the new one in 20 % (sham) vs 33 % (DCZ) of the trials (p = 0.045, q = 0.08).

  **Partially supported:** premature target touches are more frequent, while the earlier commitment to switches is only a trend.

![Figure 14 (O3).](figs/yu_dcz_O3_early_commitment.svg)

*Figure 14 (O3). Probability of touching a target while holding the lever; probability that the touched target is the chosen one (all trials) and the new one (switch trials; dashed line: 1/3).*

- **O4. DCZ decouples the speed of the response from the decision variables.** If the noradrenergic input adds a drive to act that does not depend on value, the RT should carry less information about the value of the current target and about the upcoming choice. Only the slow, reactive releases (RT ≥ 150 ms) are used, so that the extra fast releases cannot dilute these relations.
  Result:
  - On stay trials, the slope of log RT on the value of the current target (foraging model) is −0.41 under sham vs −0.18 under DCZ (p = 2×10⁻⁵, q < 0.001).
  - Under sham, slower releases go with worse choices and with switches. Under DCZ this relation is weaker: the logistic weight of log RT on choosing the best target goes from −0.39 to −0.19 (p = 2×10⁻⁵, q < 0.001), and on switching from +0.36 to +0.25 (p = 0.005, q = 0.01).
  - This is not an artefact of a compressed RT range: the SD of the reactive log RT is only 5 % smaller under DCZ (0.46 vs 0.44).

  **Supported, and the clearest new result.** Under DCZ the monkey releases the lever faster largely independently of the value of its choice and of the upcoming decision.

![Figure 15 (O4).](figs/yu_dcz_O4_vigor_decoupling.svg)

*Figure 15 (O4). Left: mean log RT by quintile of the value of the current target (stay trials, RT ≥ 150 ms). Right: per session, slope of log RT on value, and logistic weight of log RT on choosing the best target and on switching.*

- **O5. The DCZ effect is stable over the months of injections.** Repeated chemogenetic activation could lose efficacy over time (tolerance) or gain it (sensitization).
  Result: compared with the sham sessions within ±7 days, RT is shorter in 19 of 19 DCZ sessions (median −0.28 ln units), and late premature releases are more frequent in 18 of 19. There is no trend over the ~4 months, either for RT (ρ = −0.13, p = 0.59) or for late premature releases (ρ = −0.16, p = 0.50).

  **Supported:** no tolerance or sensitization.

![Figure 16 (O5).](figs/yu_dcz_O5_stability.svg)

*Figure 16 (O5). DCZ effect per DCZ session, relative to the sham sessions within ±7 days, across the months of injections (line: linear fit).*

- **O6. The DCZ effect carries over to the next day.**
  Result: sham sessions on the day after a DCZ session (n = 11, mostly Thursdays) do not differ from sham sessions after a day without injection (n = 10, mostly Tuesdays). Median log RT is 5.91 vs 5.84 (p = 0.31), and late premature releases 0.105 vs 0.117 (p = 0.38).

  **Not supported:** no next-day carryover was detected, though the groups are small and differ in weekday.

![Figure 17 (O6).](figs/yu_dcz_O6_carryover.svg)

*Figure 17 (O6). Median log RT and probability of a late premature release in sham sessions after a day without injection and after a DCZ day.*

**Synthesis.** The new results sharpen the picture of the effect on action. DCZ accelerates every lever release, both anticipatory and cue-reactive (O1). It raises premature releases only on the first hold attempt of a trial, while the correction after an error is intact (O2). Most importantly, it makes the speed of the response largely independent of the value of the choice and of the upcoming decision (O4). This is what a value-independent drive to act would produce: the activation of the noradrenergic input to the MCC energizes action initiation, while the choice computations themselves change only slightly (H8-H11). The effect is reproducible across all DCZ sessions, without tolerance (O5) and without next-day carryover (O6), so pooling the DCZ sessions is justified.

## Caveats

- A single animal: sessions are the replicates, so the effects generalize to this monkey's sessions only.
- Only H1 and H3 survive the FDR correction at q < 0.05. The cluster of effects at q ≈ 0.06-0.13 (H4-H6, H9-H11) is consistent in direction but should be confirmed, e.g. in the other monkeys or with a pre-specified test.
- *None* sessions are mostly Mondays after the weekend, so differences between none and the injection days cannot be attributed to the injection.
- The interval between injection and session start is not in the data. The within-session decline of the effect (H7) may reflect DCZ kinetics or an interaction with fatigue.
- Foraging-model parameters are fitted per session by maximum likelihood (grid search + L-BFGS-B), without hierarchical pooling. β and τ partly trade off, so the fitted switching function and the model-free loss curve are the more direct readouts.
- The original hypotheses O1-O6 were formulated after seeing the results of section (6), so they are exploratory and need confirmation in independent data.

## Sources

- Aston-Jones & Cohen (2005). An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. *Annu Rev Neurosci*.
- Bari et al. (2020). Differential attentional control mechanisms by two distinct noradrenergic coeruleo-frontal cortical pathways. *PNAS* 117:29080-29089.
- Borderies, Bornert, Gilardeau & Bouret (2020). Pharmacological evidence for the implication of noradrenaline in effort. *PLoS Biol*, doi:10.1371/journal.pbio.3000793.
- Bouret & Sara (2005). Network reset: a simplified overarching theory of locus coeruleus noradrenaline function. *Trends Neurosci* 28:574-582, doi:10.1016/j.tins.2005.09.002.
- Dubois et al. (2021). Human complex exploration strategies are enriched by noradrenaline-modulated heuristics. *eLife* 10:e59907.
- Gandaux et al. (2025). *Curr Biol* (MCC→dlPFC pathway activation with DREADDs increases engagement in a foraging task).
- Hopstaken et al. (2015). A multifaceted investigation of the link between mental fatigue and task disengagement. *Psychophysiology* 52:305-315, doi:10.1111/psyp.12339.
- Jahn et al. (2018). Dual contributions of noradrenaline to behavioural flexibility and motivation. *Psychopharmacology* 235:2687-2702, doi:10.1007/s00213-018-4963-z.
- Jepma & Nieuwenhuis (2011). Pupil diameter predicts changes in the exploration-exploitation trade-off: evidence for the adaptive gain theory. *J Cogn Neurosci*.
- Kaduk et al. (2023), as cited in the manuscript discussion (not checked here).
- Lau & Glimcher (2005). Dynamic response-by-response models of matching behavior in rhesus monkeys. *J Exp Anal Behav* 84:555-579.
- Murphy, Boonstra & Nieuwenhuis (2016). Global gain modulation generates time-dependent urgency during perceptual choice in humans. *Nat Commun* 7:13526, doi:10.1038/ncomms13526.
- Nagai et al. (2020). Deschloroclozapine, a potent and selective chemogenetic actuator enables rapid neuronal and behavioral modulations in mice and monkeys. *Nat Neurosci* 23:1157-1167, doi:10.1038/s41593-020-0661-3.
- Nassar et al. (2012). Rational regulation of learning dynamics by pupil-linked arousal systems. *Nat Neurosci* 15:1040-1046, doi:10.1038/nn.3130.
- Quilodran, Rothé & Procyk (2008). Behavioral shifts and action valuation in the anterior cingulate cortex. *Neuron* 57:314-325, doi:10.1016/j.neuron.2007.11.031.
- Robinson et al. (2008). Similar effects of the selective noradrenaline reuptake inhibitor atomoxetine on three distinct forms of impulsivity in the rat. *Neuropsychopharmacology*.
- Tervo et al. (2014). Behavioral variability through stochastic choice and its gating by anterior cingulate cortex. *Cell*.
- Ungvárszki et al. (2025). Neural dynamics reveal foraging-like computations in the frontal cortex. *bioRxiv*, doi:10.1101/2025.09.29.679289.
- Varazzani, San-Galli, Gilardeau & Bouret (2015). Noradrenaline and dopamine neurons in the reward/effort trade-off: a direct electrophysiological comparison in behaving monkeys. *J Neurosci* 35:7866-7877, doi:10.1523/JNEUROSCI.0454-15.2015.
- Warren et al. (2017). The effect of atomoxetine on random and directed exploration in humans. *PLoS ONE*, doi:10.1371/journal.pone.0176034.
- Wilson et al. (2014). Humans use directed and random exploration to solve the explore-exploit dilemma. *J Exp Psychol Gen* 143:2074-2081, doi:10.1037/a0038199.
- Yu & Dayan (2005). Uncertainty, neuromodulation, and attention. *Neuron* 46:681-692.
