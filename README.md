# SearchRL

Reinforcement Learning for Agentic Document Retrieval.

## Setup

```bash
cd /Users/ch33nchan/Desktop/exa
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Commands

### Generate Synthetic Queries (Optional)

```bash
python3 generate_queries.py --dataset nfcorpus --device cuda
```

### Train

```bash
python3 train.py \
    --dataset nfcorpus \
    --device cuda \
    --episodes 10000 \
    --lr 3e-4 \
    --batch-size 64 \
    --max-steps 5 \
    --seed 42
```

### Evaluate

```bash
python3 eval.py \
    --checkpoint checkpoints/policy_final.pt \
    --dataset nfcorpus \
    --device cuda \
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

