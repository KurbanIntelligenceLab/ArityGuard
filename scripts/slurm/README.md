# scripts/slurm/

SLURM wrappers used to run the experiments, one per experiment, numbered in run
order. Before submitting on your cluster:

- replace `YOUR_ACCOUNT` / `YOUR_QOS` and the `-p` partition in the `#SBATCH`
  lines (the runs used one A100 or H200 per job);
- submit from the repository root, e.g.
  `sbatch scripts/slurm/05b_e4b_feed_vs_pool_a100.sh Qwen/Qwen2.5-7B-Instruct`,
  or `export REPO=/path/to/repo`;
- set `CONDA_SH` if conda is not in `~/miniconda3`; the environment is
  `environment.yml`.

Drivers cache every verdict in `runs/` and flush it periodically and on SIGTERM,
so a job that hits its time limit continues when resubmitted; every job skips
seeds whose result file already exists. `submit_all.sh` and `submit_side.sh`
queue everything.
