# results/

One JSON file per experiment, agent and seed (`<EXP>_<agent>_seed<s>.json`),
written by the drivers in `code/arq/experiments.py`. `make tables` reads these
files and rebuilds every table in the paper; no number in the paper was typed by
hand. Experiment codes and where they appear:

| Files | Experiment | Paper |
|---|---|---|
| `E4b_*`, `robustrag/E4b_*` | feed against pool (delivered-feed certificate vs pool resampling) | RQ1, Table 2 |
| `E2_*`, `synthetic_entities/E2_*` | arity validation on known-arity pools; invented entity names | RQ2, Table 3 |
| `relevance_filtered/E1_*`, `E3_*` | arity probe on CoPHEME; curve prediction | RQ3, Table 4 |
| `E4_*` | montage attack data, ARQ against baselines | RQ4, Table 5 |
| `E5b_*`, `E5_*` | ablations; stress test (adding vs evicting coalitions) | RQ5, Table 6 |
| `E7_*` | misspecified share (replay of the RQ1 votes) | supplementary |
| `COST_*` | measured calls, tokens and GPU seconds | supplementary |
| `side/third_model/` | Mistral-7B-Instruct-v0.3: feed against pool, arity validation | supplementary |
| `side/coalition/COAL_*` | real-text coalition (reposted montage posts) | supplementary |
| `E6_seed0.json`, `e6_examples_*` | verbatim cases | supplementary |

`side/` holds runs that were first done alongside the main experiments and
were then added to the supplementary.
