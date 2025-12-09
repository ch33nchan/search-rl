# SearchRL

RL-driven agentic search with learned query reformulation. This project explores training policies for multi-step retrieval instead of single-pass embedding lookup, optimizing retrieval quality through sequential decision-making.

**Core Idea**: Instead of encode-once-retrieve, an agent iteratively refines queries and searches until it finds optimal results. Trained with PPO on dense retrieval + LLM-as-judge rewards.

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

## Problem

Single-pass dense retrieval fails on ambiguous, underspecified, or complex queries. Traditional systems encode query → similarity search → return top-k. No mechanism to refine, explore alternatives, or adapt based on initial results.

## Solution

Treat retrieval as an MDP. Agent observes (query, retrieved docs) and chooses actions:

- **Search** (A0): Execute dense retrieval with current query
- **Narrow** (A1): LLM reformulates query to be more specific
- **Broaden** (A2): LLM reformulates query to be more general
- **Terminate** (A3): Commit to current results

Policy network (GRU over search trajectory) trained with PPO to maximize nDCG@10 from LLM-judged relevance. Learns when to reformulate vs when single-pass suffices.

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

## Technical Highlights

**Action Masking**: Prevents degenerate policies (must search before terminate/reformulate).

**Curriculum Learning**: Pre-compute query difficulty, oversample hard queries (low single-pass recall) after warmup.

**Multi-GPU Pipeline**: Reward model (GPU0) and reformulator (GPU1) run in parallel during rollouts.

**Reward Shaping**: Step penalties, empty termination penalties, action-masking to guide exploration.

**Evaluation Framework**: Policy vs single-pass vs always-reformulate baselines, stratified by query difficulty.

## Key Results

**FIQA (Financial Q&A)**: Agent learns to reformulate on ambiguous queries, outperforms single-pass on hard queries (recall < 0.6).

**NFCorpus (Medical)**: Agent learns single-pass is optimal, never uses reformulation. Reformulation actually hurts performance (-0.07 nDCG). Policy correctly adapts to dataset characteristics.

**Takeaway**: RL agent learns dataset-specific optimal search strategies rather than blindly applying reformulation heuristics.

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

## Future Directions

- Scale to web-scale indices (billions of documents)
- Replace LLM reformulator with learned query encoder
- End-to-end training: jointly train retriever + policy
- Multi-hop reasoning over retrieved documents
- RLAIF pipeline for preference learning on search quality

## Technical Details

Full experiment history, hyperparameter sweeps, ablations, and failure modes documented in `project_status.md`.

## Stack

Python 3.10+ | PyTorch 2.1+ | CUDA 11.8+ | Transformers | Sentence-Transformers | FAISS | BEIR
