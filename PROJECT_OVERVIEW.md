# SearchRL: Project Overview

## What Is This?

SearchRL is a reinforcement learning system that learns how to search for documents. Instead of doing a single embedding lookup and returning results, an agent makes multiple decisions to iteratively improve retrieval quality.

## The Problem

Traditional retrieval systems:
1. Encode the query once
2. Compute similarity against document index
3. Return top-k results

This fails when queries are:
- **Ambiguous**: "recent advances in protein folding" - recent how? what kind of advances?
- **Underspecified**: Missing context that would help find better documents
- **Complex**: Requiring information from multiple perspectives or documents

Single-pass retrieval either gets lucky or fails entirely. There's no mechanism to refine, explore, or adapt.

## The Solution

Reformulate retrieval as a **sequential decision problem**. The agent:

1. **Observes**: Current query + retrieved documents so far
2. **Acts**: Chooses among 4 actions:
   - **Search**: Execute retrieval with current query
   - **Narrow**: Reformulate query to be more specific (using LLM)
   - **Broad**: Reformulate query to be more general (using LLM)
   - **Terminate**: Commit to current results and end episode
3. **Receives Reward**: nDCG@10 computed from LLM-as-judge relevance scores

The agent learns a policy (via PPO) that maps search trajectories to actions, optimizing for retrieval quality over the full episode.

## Architecture

### Components

1. **Environment** (`src/environment.py`)
   - Gymnasium-compatible search game
   - Manages episode state, actions, rewards
   - Tracks search trajectory

2. **Retriever** (`src/retriever.py`)
   - SentenceTransformers encoder (all-MiniLM-L6-v2)
   - FAISS index for fast similarity search
   - Black-box API - agent doesn't modify it, just uses it

3. **Policy Network** (`src/policy.py`)
   - GRU encoder over search trajectory (query embeddings + actions + result embeddings)
   - MLP head outputs action logits and value estimate
   - ~500k parameters, lightweight

4. **Reward Model** (`src/reward.py`)
   - Qwen-3B/1.5B LLM acts as relevance judge
   - Scores query-document pairs 0-10
   - Computes nDCG@10 from scores

5. **Reformulator** (`src/reformulator.py`)
   - Qwen LLM rewrites queries (narrow/broad)
   - Conditioned on current retrieved documents
   - Parameterized generation, not templates

6. **PPO Trainer** (`src/ppo.py`)
   - Proximal Policy Optimization
   - Generalized Advantage Estimation
   - Standard RL training loop

### Training Flow

```
1. Sample query from synthetic query bank
2. Agent takes actions until termination (max 5 steps)
3. Compute nDCG@10 reward from LLM judgments
4. Store trajectory in rollout buffer
5. After batch_size episodes, compute GAE advantages
6. Run PPO update on policy network
7. Repeat
```

### Synthetic Query Generation

Training data generated from corpus using LLM:
- **Single-doc queries**: "What question would this document uniquely answer?"
- **Multi-doc queries**: Require information from 2-3 related documents
- **Adversarial queries**: Keyword matches fail, semantic understanding needed

Each query paired with ground-truth relevant documents for nDCG computation.

## Key Design Decisions

1. **Modular**: Retriever is black-box, policy can be swapped/retrained independently
2. **LLM-as-judge**: No human labels needed, scales to millions of episodes
3. **Trajectory encoding**: GRU over sequence, not just current state
4. **Adaptive compute**: Agent learns when to search vs. reformulate vs. terminate
5. **Portable**: Works with any retrieval backend (SentenceTransformers, FAISS, etc.)

## File Structure

```
src/
├── config.py          # Configuration dataclasses
├── retriever.py       # FAISS + SentenceTransformers retrieval
├── reward.py          # LLM-as-judge relevance scoring
├── reformulator.py    # LLM-based query rewriting
├── query_gen.py       # Synthetic query generation
├── policy.py          # GRU + MLP policy network
├── environment.py     # Gymnasium search environment
├── ppo.py             # PPO training algorithm
├── evaluation.py      # Evaluation and baselines
├── data.py            # BEIR dataset loading
└── utils.py           # Device detection, Metal optimizations

train.py               # Main training script
eval.py                # Evaluation script
generate_queries.py    # Query generation script
benchmark.py           # Performance benchmarking
```

## Getting Started

### Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Benchmark (test your hardware)
python3 benchmark.py --device cuda  # or mps

# Generate queries (one-time, ~10-20 min)
python3 generate_queries.py --dataset nfcorpus --device cuda

# Train (3-6 hours on H100, 8-14 hours on M4 Pro)
python3 train.py --dataset nfcorpus --device cuda --episodes 10000

# Monitor
tensorboard --logdir runs

# Evaluate
python3 eval.py --checkpoint checkpoints/policy_final.pt --dataset nfcorpus
```

### Key Parameters

- `--episodes`: Number of training episodes (10000 default)
- `--batch-size`: PPO batch size (64 for CUDA, 32 for MPS)
- `--max-steps`: Max actions per episode (5 default)
- `--lr`: Learning rate (3e-4 default)
- `--device`: cuda, mps, or cpu (auto-detects)

## Evaluation

Compares three methods:
1. **Single-pass**: Embed query once, return top-k
2. **Heuristic agent**: Fixed policy (search → narrow → search → broad → terminate)
3. **RL agent**: Learned policy

Metrics:
- **nDCG@10**: Normalized discounted cumulative gain
- **MRR@10**: Mean reciprocal rank
- **Avg Queries/Episode**: Compute cost

The RL agent should learn to:
- Terminate quickly on easy queries (1-2 steps)
- Use more steps on hard queries (4-5 steps)
- Outperform heuristic on hardest queries

## Current Status

- ✅ Full implementation complete
- ✅ Metal (MPS) optimizations for Apple Silicon
- ✅ CUDA support for H100
- ✅ Synthetic query generation
- ✅ PPO training loop
- ✅ Evaluation framework
- 🚧 Training in progress (baseline experiments)

## Next Steps / TODOs

1. **Hyperparameter tuning**: Learning rate, batch size, GAE lambda
2. **Architecture experiments**: Different GRU sizes, attention mechanisms
3. **Reward shaping**: Intermediate rewards for good reformulations
4. **Curriculum learning**: Start with easy queries, gradually increase difficulty
5. **Multi-task**: Train on multiple BEIR datasets simultaneously
6. **Ablation studies**: What if we remove reformulation? What if we use fixed templates?

## Key Papers/Concepts

- **RLAIF**: Reinforcement Learning from AI Feedback (no human labels)
- **PPO**: Proximal Policy Optimization (stable RL algorithm)
- **nDCG**: Standard IR metric for ranking quality
- **BEIR**: Benchmark for IR evaluation datasets
- **LLM-as-judge**: Using language models to score outputs

## Questions?

- **Why GRU?** Captures sequential dependencies in search trajectory
- **Why 4 actions?** Small discrete space, easy to learn, interpretable
- **Why LLM reformulation?** Flexible, parameterized, not limited to templates
- **Why nDCG@10?** Standard IR metric, rewards ranking quality not just top-1
- **Why synthetic queries?** Scales to millions of examples, no annotation cost

## Contact / Contributing

See main README.md for detailed usage instructions.

