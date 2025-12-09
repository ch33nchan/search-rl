# SearchRL

> Reinforcement learning for iterative document retrieval with query reformulation.

An RL agent learns to search by making sequential decisions: search, narrow query, broaden query, or terminate. Trained using PPO on BEIR datasets to optimize nDCG@10.

## Quick Start

```bash
# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Train on FIQA dataset
python train.py --dataset fiqa --device cuda --episodes 10000

# Evaluate trained policy
python eval.py --checkpoint checkpoints/policy_final.pt --dataset fiqa --num-queries 200

# Monitor training
tensorboard --logdir runs
```

## Overview

Traditional retrieval does single-pass search and returns results. SearchRL treats retrieval as a sequential decision problem where an agent can:

- **Search** (A0): Execute retrieval with current query
- **Narrow** (A1): Reformulate query to be more specific
- **Broaden** (A2): Reformulate query to be more general  
- **Terminate** (A3): End episode and compute reward

The agent uses a GRU-based policy network trained with PPO, learning from LLM-judged relevance scores (nDCG@10).

## Architecture

```
src/
├── config.py          # Configuration dataclasses
├── environment.py     # Gymnasium search environment
├── policy.py          # GRU policy network (actor-critic)
├── ppo.py             # PPO training algorithm
├── retriever.py       # Dense retrieval (FAISS + SentenceTransformers)
├── reformulator.py    # LLM-based query rewriting
├── reward.py          # LLM-as-judge relevance scoring
├── evaluation.py      # Policy evaluation utilities
├── data.py            # BEIR dataset loading
├── query_gen.py       # Synthetic query generation
└── utils.py           # Helper functions
```

## Training Commands

### Basic Training
```bash
python train.py \
    --dataset fiqa \
    --device cuda \
    --episodes 10000 \
    --lr 3e-4 \
    --batch-size 64
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --dataset fiqa \
    --device cuda \
    --episodes 10000 \
    --reward-gpu 0 \
    --reformulator-gpu 1
```

### Curriculum Learning
```bash
python train.py \
    --dataset fiqa \
    --device cuda \
    --episodes 10000 \
    --curriculum \
    --hard-threshold 0.6 \
    --hard-ratio 0.7
```

## Evaluation

```bash
# Evaluate trained policy
python eval.py \
    --checkpoint checkpoints/policy_final.pt \
    --dataset fiqa \
    --device cuda \
    --num-queries 200
```

Outputs:
- Policy nDCG@10 and action distribution
- Single-pass baseline comparison
- Heuristic baseline (always reformulate) comparison
- Breakdown by query difficulty

## Datasets

Supported BEIR datasets:
- **fiqa** - Financial Q&A (recommended, more ambiguous)
- **nfcorpus** - Medical literature (simpler queries)
- **scifact** - Scientific fact verification
- **arguana** - Argumentative texts
- **scidocs** - Citation prediction

## Key Features

### Action Masking
Agent must search before terminating or reformulating, preventing degenerate policies.

### Curriculum Learning
Focus training on hard queries (low single-pass recall) to learn when reformulation helps.

### Multi-GPU Support
Run reward model and reformulator on separate GPUs for 2x speedup.

### Device Support
Automatic detection: `cuda > mps > cpu`

## Results

Training on FIQA (10k episodes):
- Agent learns to use narrow/broaden actions for ambiguous queries
- Outperforms single-pass baseline on hard queries
- Properly learns when NOT to reformulate

Training on NFCorpus:
- Agent correctly learns single-pass is optimal
- Reformulation hurts performance on this dataset
- Validates that RL is learning meaningful policy

## Configuration

Key hyperparameters in `src/config.py`:

```python
# PPO
lr = 3e-4
batch_size = 64
epochs = 4
clip_epsilon = 0.2
entropy_coef = 0.05

# Environment
max_steps = 5
step_penalty = 0.02
top_k = 10

# Curriculum
hard_query_threshold = 0.6
hard_query_ratio = 0.7
warmup_episodes = 1000
```

## Project Status

See `project_status.md` for detailed experiment history and results.

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (optional, for GPU)
- Transformers, Sentence-Transformers, FAISS, BEIR

## License

MIT
