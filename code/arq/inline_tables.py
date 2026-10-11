"""
Copy generated table macros INTO a .tex file, between two marker lines, so
the paper is one self-contained file (no \\input{generated/...}). Run by
`make tables` right after `python -m arq.report`, so every number in the
block still comes from arq.report and nothing is typed by hand.

    python -m arq.inline_tables paper/arq_main.tex paper/generated \\
        table_bound table_arity ...

Everything between the markers is replaced on each run; text outside them is
never touched. The script fails if the markers are missing or duplicated, or
if a listed table file does not exist, rather than writing a partial block.
"""
from __future__ import annotations

import os
import sys

BEGIN = "%%% BEGIN GENERATED TABLES"
END = "%%% END GENERATED TABLES"


def inline(tex_path: str, gen_dir: str, names) -> None:
    with open(tex_path) as fh:
        lines = fh.read().split("\n")
    b = [i for i, l in enumerate(lines) if l.startswith(BEGIN)]
    e = [i for i, l in enumerate(lines) if l.startswith(END)]
    if len(b) != 1 or len(e) != 1 or b[0] >= e[0]:
        raise SystemExit(f"{tex_path}: need exactly one '{BEGIN}' line before one '{END}' line")
    body = [f"{BEGIN} (written by `make tables` from arq.report; do not edit by hand)"]
    for n in names:
        path = os.path.join(gen_dir, n + ".tex")
        if not os.path.exists(path):
            raise SystemExit(f"missing generated table {path}; run arq.report first")
        with open(path) as fh:
            body.append(f"% --- {n}")
            body.extend(fh.read().rstrip("\n").split("\n"))
    out = lines[:b[0]] + body + lines[e[0]:]
    with open(tex_path, "w") as fh:
        fh.write("\n".join(out))
    print(f"inlined {len(names)} table(s) into {tex_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    inline(sys.argv[1], sys.argv[2], sys.argv[3:])
