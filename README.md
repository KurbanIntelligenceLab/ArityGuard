# ArityGuard: Certified Answers for LLM Agents Facing Coalitions That Fabricate Nothing

Code and results for the paper (arXiv: TODO-add-link).

LLM agents increasingly answer questions from evidence that other agents
publish. A coalition of such agents can mislead them without writing anything
false, simply by choosing which true posts are easy to find. **ArityGuard** lets
an agent decide, before it acts, whether an answer the coalition could not have
caused is possible, and returns a certified answer when it is. It rests on one
quantity, *reading arity*: how many evidence items the agent must read together.
The task sets a minimum (a group too small to answer abstains); the coalition's
share of the evidence sets a maximum (larger groups are likelier to contain one
of its items). ArityGuard measures arity on the agent's own pool, checks a
closed-form bound on whether minimum and maximum are compatible, and runs
**ARQ** (arity-restricted quorum), a vote over small groups resampled from the
pool that certifies a verdict only if it would win without any coalition vote.
What ARQ certifies is independence from the coalition, not truth.

## Reproduce the paper's tables (no GPU, about a minute)

```bash
conda env create -f environment.yml && conda activate arq_aamas27   # or: pip install -r requirements.txt
make verify    # 22 exhaustive / Monte Carlo checks of the certificate, no model
make smoke     # the whole pipeline end to end on a mock agent
make tables    # every table of the paper, from results/, into tables/*.tex
```

`results/` holds the raw output of every run behind the paper, so these steps
need neither a GPU nor the datasets. `make smoke` prints two FALSIFICATION
alarms; they are expected with the mock agent.

## The method in code

| File | What |
|---|---|
| `code/arq/certificate.py` | exact clean-group probability, the contamination bound D̄, `certify()`, the arity bound ρ*(τ, q), Wilson intervals, τ |
| `code/arq/pool.py` | evidence pools with nested coalition sets; group sampling; adding and evicting coalitions |
| `code/arq/agent.py` | the verdict prompt (TRUE / FALSE / ABSTAIN), local Hugging Face and API agents, strict parsing, a cache keyed on the rendered prompt |
| `code/arq/baselines.py` | undefended agent, RobustRAG-style feed partition, majority vote, LLM judge, influence detector |
| `code/arq/experiments.py` | the experiment drivers |
| `code/arq/report.py` | builds every table from `results/` |
| `code/tests/test_certificate.py` | the verification suite |

The certificate on its own, run from `code/`. Collect one verdict per group of
`t` items resampled from a pool of `n` items (any agent will do), then:

```python
from arq.certificate import certify, rho_star
votes = ["FALSE"] * 150 + ["ABSTAIN"] * 40 + ["TRUE"] * 10   # G = 200 groups of t = 2
r = certify(votes, n=300, rho=0.05, t=2, eta=0.01)            # rho: assumed coalition share
print(r.certified, r.verdict, r.margin, r.threshold)           # True FALSE 110 30
print(rho_star(tau=2, q=0.9))                                  # 0.155: largest tolerable share
```

Use `adversary="displace"` when the coalition can evict authentic items (for
example behind a fixed top-K retriever); the threshold then doubles.

## Re-running the experiments (GPU)

`make data` fetches HotpotQA, 2WikiMultihopQA, the CoPHEME / montage-attack
repository and RobustRAG's data at pinned versions, and installs our generated
attack plans (`data/generated_attack_plans/`). Agents: Qwen2.5-7B-Instruct,
Qwen2.5-14B-Instruct and Mistral-7B-Instruct-v0.3, greedy decoding; they
download on first use (`python scripts/download/fetch_models.py` pre-fetches
them). Each driver caches every verdict in `runs/` and can be resumed.

| Paper | Driver (`scripts/`) | SLURM wrapper (`scripts/slurm/`) | Results | GPU time, approx. |
|---|---|---|---|---|
| RQ1, Table 2 (upper) | `run_e4b.py --model-id M --seed s` | `05b_*` (seed 0), `05c_*` (seeds 1-4) | `results/E4b_*` | 1.5-3 h per seed |
| RQ1, Table 2 (lower), RobustRAG data | `run_e4c_robustrag.py` | `11_*` | `results/robustrag/` | 3-6 h |
| RQ2, Table 3 | `run_e2.py`, `run_e2_synth.py` | `03_*`, `10_*`, `12_* e2` | `results/E2_*`, `results/synthetic_entities/` | ~13 h per seed (7B) |
| RQ3, Table 4 | `run_e1.py --relevance-filter`, `run_e3.py` | `02b_*`, `04_*`, `12_* e1/e3` | `results/relevance_filtered/`, `results/E3_*` | 4-24 h per seed |
| RQ4, Table 5 | `run_e4.py --relevance-filter` | `05_*`, `12_* e4` | `results/E4_*` | a few hours per seed |
| RQ5, Table 6 | `run_e5b.py`, `run_e5.py` | `08_*`, `06_*`, `12_* e5b/e5` | `results/E5b_*`, `results/E5_*` | 3-6 h; E5 up to 2.5 days |
| Supp., cost | `run_cost.py` | `09_*` | `results/COST_*` | 2-4 h |
| Supp., misspecified share | `run_e7.py --cache-only` | none (CPU) | `results/E7_*` | needs the RQ1 vote caches |
| Supp., third model family | `run_e4b.py`, `run_e2.py`, `run_e2_synth.py` | `13_*` | `results/side/third_model/` | ~1.75 h per seed |
| Supp., real-text coalition | `run_side_coalition.py` | `14_*` | `results/side/coalition/` | 1.5-3 h per seed |
| Supp., paired tests | none | none | the RQ1 results | none |
| Supp., verbatim cases | `run_e6.py` | `07_*`, `07b_e6_*` | `results/E6_seed0.json`, `results/e6_examples_*` | under an hour |

The misspecified-share sweep replays the votes cast in RQ1 rather than calling
a model; it needs the RQ1 vote caches (`runs/e4b_feed_vs_pool/`), which a GPU
rerun of RQ1 recreates. Experiment codes (E1-E7) in file names are internal;
`results/README.md` maps them to the paper.

## Data and responsible use

The datasets are public and are not redistributed here (`data/README.md`). The
montage attack plans in `data/generated_attack_plans/` were generated with an
open model for three CoPHEME events that the attack's authors did not release
plans for; they are misinformation-style posts about real events, made to
evaluate defenses, and should be used only for that purpose.

## License

TODO: choose a license before publishing.

## Citation

TODO: add the BibTeX entry once the arXiv version is out.
