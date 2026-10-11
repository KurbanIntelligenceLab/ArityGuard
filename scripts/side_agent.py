"""
Agent helpers for the SIDE experiments (results/side/, not in the paper).

make_agent(model_id, chat) returns the production LocalAgent, or, with
chat=True, ChatLocalAgent: the same prompt (Agent._render, unchanged) wrapped
in the tokenizer's chat template before greedy decoding. The Qwen runs in the
paper used the raw prompt, so raw is the default and keeps the protocol
identical; chat is the fallback for a model that does not answer the raw
prompt with a verdict word.

probe(agent, prompts) runs a handful of real verdicts and reports how many
parse, so a job can pick a mode before spending hours (see 13_*.sh).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from arq.agent import LocalAgent  # noqa: E402


class ChatLocalAgent(LocalAgent):
    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, **kw)
        # Distinct cache identity: chat-wrapped outputs must never be mixed
        # with raw-prompt outputs of the same model.
        self.name = f"{model_id}::chat"

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch  # noqa
        ids = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def make_agent(model_id: str, chat: bool = False):
    return ChatLocalAgent(model_id) if chat else LocalAgent(model_id)


def probe(agent, cases) -> int:
    """cases: list of (items, claim). Prints each raw answer; returns the
    number that fail to parse. Calls the model directly (no cache)."""
    agent._load()
    bad = 0
    for items, claim in cases:
        raw = agent._generate(agent._render(items, claim), agent.max_new_tokens + 8)
        try:
            v = agent._parse(raw)
        except ValueError:
            v, bad = "UNPARSEABLE", bad + 1
        print(f"  probe: {v:11s} <- {raw.strip()[:60]!r}")
    print(f"probe: {len(cases) - bad}/{len(cases)} parsed")
    return bad
