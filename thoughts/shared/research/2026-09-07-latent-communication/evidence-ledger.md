# Internal research ledger

Access date: 2026-09-07. Scope: user's four questions, current local KVComm, static audit plus primary research; no implementation or benchmark run authorized. Audience: developer designing experiments. Geography immaterial; hardware constraint supplied by user: four RTX 3080 cards.

Plan: scope/primary-source discovery completed; local audit completed; targeted disconfirmation completed; synthesis completed; artifact structural verification completed. update_plan tool unavailable after metadata search; progress communicated in conversation.

## Claim-gap matrix

| Claim family | Evidence | Confidence / caveat | Remaining gap |
| --- | --- | --- | --- |
| TextMAS dominates simple HotpotQA | Official multi-hop definition; local oracle sentence loader | High on input protocol; no dominance evidence | Matched current-prompt predictions |
| Latent universally wins reasoning | Latest LatentMAS Table 1 includes opposite GSM8K rankings by model; Coconut counterexample | High rejection of universal claim | Local model and two-agent ranking |
| Selected KV lowers practical cost | Local copy-before-selection, layer-zero retention, slice views | High static evidence; runtime cost conditional | GPU profiler and measured bytes |
| Prompts are fair/effective | Shared prompt core, asymmetric MedQA options, thinking budgets and different truncation | High static; optimality unproven | Budget/placement ablation |
| Metric labels trustworthy | Base/custom evaluator mismatch; task-specific metric code | High static; not every primary value invalid | Regression tests and record-level audit |
| Cross-model cache compatibility | Local layer mapping; C2C learned fuser | High that arbitrary compatibility is unproven | Exact supported model-pair testing |
| KVComm paper directly compares current TextMAS | v1 sender is query-blind; local QA sender query-aware | High mismatch | Final paper full text inaccessible |
| New combined method wins | No matched current results inspected | Unestablished | Controlled held-out benchmark |

## Primary-source provenance

- Latent Collaboration in Multi-Agent Systems; Jiaru Zou et al.; arXiv v4 2026-08-03; https://arxiv.org/html/2511.20639v4 . Parent browser read turn19view0. Table 1 independently spot-checked: sequential GSM8K 4B text89.8 latent88.2, 8B text92.3 latent93.8. Latest version supersedes v1 for report. 200-word attribution budget; concise derived content only.
- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing; Shi et al.; arXiv v1 October 2025; https://arxiv.org/html/2510.03346v1 . Parent turn19view1 / turn20view0; same-base restriction, query-blind sender, FLOPs interpretation. Final OpenReview F7rUng23nw challenged, not treated as full-text reviewed. Agent also verified official ICLR poster https://iclr.cc/virtual/2026/poster/10010626 ; no final-text numerical claims.
- Training Large Language Models to Reason in a Continuous Latent Space; Shibo Hao et al.; arXiv v3 (version-specific, exact update date not used); https://arxiv.org/html/2412.06769v3 . Parent turn19view2 / turn20view1. Table 1 and section 5: GSM8K CoT42.9 vs Coconut34.1, curriculum distinction. Report omits figures and does not extrapolate to training-free multi-agent.
- Cache-to-Cache: Direct Semantic Communication Between Large Language Models; arXiv v1 October 2025; https://arxiv.org/html/2510.03215v1 . Parent turn19view3 / turn20view2. Learned fuser and training regime, not plug-and-play unaligned KV. No numerical speedup in report.
- HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering; Zhilin Yang et al.; EMNLP2018; https://hotpotqa.github.io/ . Parent turn21view0 and earlier read. Official ten-paragraph distractor plus support prediction; local task differs.
- LongBench metrics.py; THUDM official repository, live main read 2026-09-07; https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/metrics.py . Parent turn21view1. Counter token F1 and normalization. No pinned commit asserted.
- Lost in the Middle: How Language Models Use Long Contexts; Nelson F. Liu et al.; arXiv2023 / TACL2024; https://arxiv.org/abs/2307.03172 . Parent turn21view2. Position sensitivity only; does not prove proposed prompt order better.
- KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems; Ye et al.; NeurIPS2025; https://papers.nips.cc/paper_files/paper/2025/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html . Agent provenance naming disambiguation only, not merged with Shi selective KVComm.
- Official LatentMAS implementation and README, Gen-Verse, live main; https://github.com/Gen-Verse/LatentMAS ; agent read HF vs vLLM and role setup. Not used as a loophole to expand same-paper attribution.

## Local evidence and access boundaries

Read prompts_latent.py, eval_latent.py, com_latent.py, models_latent.py, models.py, task loaders, dataloader/base_evaluator.py, utils/f1.py, utils/metric_tools.py, sweep_latent.sh, plan and docs/problem_v1.md during research. Exact file links embedded adjacent to report claims.

Spot checks repeated by coordinator: Hotpot support sentences, MuSiQue support paragraphs; .to before selection and attention_mask omission; TextMAS exception skip; MedQA A lacks options; task metric overrides; primary/secondary metric counters; CausalLM prefill; mode3 initialization without selection calibration in that branch.

docs/problem_v1.md cites EXPERIMENT_RESULTS.md, but that root-level file was not found in final spot-check. Report only attributes historical observation to available summary document and explicitly does not authenticate original run artifacts. Some diagnostic guesses for nonexistent paths base.py/utils.py/arc.py were corrected or dropped; no claims based on these missing files.

## Searches and stopping rule

First wave: original LatentMAS and KVComm papers/repos, official task definitions and metrics; local input/prompt/eval/cache path audit. Two bounded independent research agents handled LatentMAS evidence and KVComm/adjacent papers as required by skill for substantial independent lanes.

Second wave: latest LatentMAS version, disconfirming task/model cases, sender query visibility, FLOPs vs latency, Coconut reasoning counterexamples, learned cross-model C2C, local measurement and prompt confounds. Parent reopened consequential external sources to establish visible provenance.

Stop: all four question families have primary support or an explicit gap; remaining superiority/optimality claims require local experiments, not more general literature. Final KVComm text access failure bounded without repeated retries. No exhaustive-literature-review claim.

## Artifact

Canonical report-source.md created once after substantive synthesis, then one scoped paragraph refinement. Single user-facing research-report.html derived from canonical source with UTF-8, responsive layout, tables and adjacent descriptive source links. Internal files not linked in final user response. Structural QA passed: file 31936 bytes, 11 h2 sections, four opening/closing tables, 37 links, zero broken local targets, closing HTML present, zero Unicode replacement characters. Visual browser review not performed; disclosed in handoff. Git status unavailable because workspace root is not a Git repository; this does not affect read-back verification of newly created artifacts.
