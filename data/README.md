# data/

Adapters that turn each dataset into evidence pools (`*_adapter.py`,
`synthetic_entities.py`) and the montage attack plans we generated
(`generated_attack_plans/`). The datasets themselves are public and are fetched
by `make data` (`scripts/download/`), pinned and checksum-verified:

| Path | Source |
|---|---|
| `hotpotqa/hotpotqa_distractor_validation.jsonl` | Hugging Face `hotpotqa/hotpot_qa`, distractor, validation |
| `2wikimultihopqa/2wiki_validation.jsonl` | Hugging Face `voidful/2WikiMultihopQA`, validation |
| `Lying_with_Truth/` | github.com/CharlesJW222/Lying_with_Truth @ 21183d3 (CoPHEME and the montage attack) |
| `RobustRAG/` | github.com/inspire-group/RobustRAG @ 9bc35b2 (RealtimeQA, NQ) |

The attack paper ships montage plans for three of the six CoPHEME events. The
other three (germanwings-crash, putinmissing, sydneysiege; 410 plans) were
generated for this work with `scripts/generate_montage_batch.py` and
Qwen2.5-7B-Instruct; `make data` copies them into the clone, where
`copheme_adapter.py` reads them. They are attack material produced for
evaluating defenses; please use them only for research on misinformation
defenses.
