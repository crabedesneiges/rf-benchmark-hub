# SLURM job scripts

Batch scripts that run RF-Benchmark-Hub work on a SLURM cluster: dataset
download/prepare (CPU), baseline training and foundation-model evaluation (GPU),
and environment/setup probes. They were written for the **Dalia (IDRIS)** cluster
(ARM `aarch64` GB200 nodes, Lustre filesystem) but the cluster-specific paths are
**not hardcoded** — every script reads them from environment variables with a
sensible default, so you can adapt them to another cluster without editing the
scripts.

## Configuration variables

Each script exposes a "Portable config" block at the top. Set these in your
environment (or `export` them in your shell profile) to point the scripts at your
own paths; unset variables fall back to the Dalia-style defaults shown below.

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORK` | **Required.** Lustre work root (code, envs, cache, logs live under it). On Dalia this is pre-set by the cluster. | *(none — the script aborts if unset)* |
| `RFBENCH_REPO` | Repo/worktree checkout to run. Most training scripts also honor `SLURM_SUBMIT_DIR` so they run the code of the worktree you submit from. | `$WORK/projets/rf-benchmark-hub` (some scripts default to a task-specific worktree — see the script header) |
| `RFBENCH_VENV_CPU` | CPU virtualenv, installed with `.[dev,data]` (numpy/h5py/requests). | `$WORK/envs/rfbench-arm` |
| `RFBENCH_VENV_GPU` | GPU virtualenv, installed with `.[dev,data,tasks,torch]`. | `$WORK/envs/rfbench-arm-gpu` |
| `RFBENCH_VENV_DETECTION` | Detection virtualenv, installed with `.[dev,detection]` (torchsig). | `$WORK/envs/rfbench-arm-detection` |
| `RFBENCH_UV` | `uv` binary for the compute-node architecture. | `$WORK/envs/uv-arm/uv` |
| `RFBENCH_CACHE` | Dataset cache root (raw data lives here, never in git). | `$WORK/data/rfbench_cache` |

Example — run a baseline on a different cluster layout:

```bash
export WORK=/scratch/$USER/rfbench
export RFBENCH_VENV_GPU=$WORK/venvs/gpu
export RFBENCH_CACHE=/data/rf_cache
mkdir -p logs
sbatch slurm/train_baseline_arm.sh mcldnn 50 42
```

## Logs

SLURM job stdout/stderr go to `logs/<name>_%j.out` and `logs/<name>_%j.err`
**relative to the directory you submit from** (`%j` is the job id). Create the
directory first:

```bash
mkdir -p logs
sbatch slurm/<script>.sh ...
```

or override per-invocation with `sbatch --output=/path/%j.out --error=/path/%j.err ...`.

Training/eval scripts additionally stage their real artifacts under `$WORK`:

- `result.json` staging → `$WORK/logs/multiseed/<task>/...`
- checkpoints → `$WORK/checkpoints/multiseed/...`

## Cluster notes (Dalia / IDRIS)

- Compute nodes are ARM `aarch64` with NVIDIA GB200 GPUs; the login node is Intel
  `x86_64`. **Build venvs on a compute node** — x86_64 wheels are not compatible.
  Hence the absolute `RFBENCH_UV` path: `uv` on `$PATH` may resolve to the login
  node's x86_64 binary.
- The cluster is effectively single-partition (GB200/ARM only), so the scripts set
  no `--constraint`. Adjust the `#SBATCH` partition/constraint directives for your
  site.
- `--mem` is intentionally omitted (Dalia auto-allocates RAM proportional to the
  requested cores).
- Many scripts prepend `$REPO` to `PYTHONPATH` so `import rfbench` resolves to the
  submitted worktree rather than whatever editable install the venv registered.
