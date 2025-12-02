# SearchRL

Reinforcement Learning for Agentic Document Retrieval.

## Setup

```bash
cd /Users/ch33nchan/Desktop/exa
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Metal (MPS) Optimization for Apple Silicon

The code automatically detects and optimizes for Metal Performance Shaders on M4 Pro:
- Uses smaller models (1.5B instead of 3B) for faster inference
- Optimized batch sizes for Metal
- Automatic device detection (mps > cuda > cpu)

**M4 Pro Training Time Estimate:**
- Per episode: ~2-5 seconds
- 10k episodes: ~6-14 hours (depending on query complexity)
- Run `python3 benchmark.py` for precise timing on your system

## Commands

### Benchmark Device Performance

```bash
python3 benchmark.py --device mps
```

### Generate Synthetic Queries (Optional)

```bash
python3 generate_queries.py --dataset nfcorpus --device mps
```

### Train

```bash
python3 train.py \
    --dataset nfcorpus \
    --device mps \
    --episodes 10000 \
    --lr 3e-4 \
    --batch-size 64 \
    --max-steps 5 \
    --seed 42
```

Device auto-detects if not specified (uses mps on Apple Silicon).

### Evaluate

```bash
python3 eval.py \
    --checkpoint checkpoints/policy_final.pt \
    --dataset nfcorpus \
    --device mps \
    --num-queries 200
```

### Monitor Training

```bash
tensorboard --logdir runs
```

## Datasets

Supported BEIR datasets:
- nfcorpus
- scifact
- fiqa
- arguana
- scidocs

## Actions

- 0: Search - Execute retrieval with current query
- 1: Narrow - Reformulate query to be more specific
- 2: Broad - Reformulate query to be more general
- 3: Terminate - End episode and compute reward

