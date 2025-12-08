# SearchRL Project Status

**Last Updated**: December 8, 2025

## Project Overview

SearchRL is a reinforcement learning system that learns how to iteratively search and reformulate queries to improve document retrieval. Instead of single-pass retrieval, an RL agent makes sequential decisions (search, narrow, broaden, terminate) to optimize nDCG@10.

## Current Status: Training on FIQA Dataset

Training is currently running on the FIQA (Financial Q&A) dataset after experiments on NFCorpus showed that reformulation doesn't help on that dataset.

---

## Experiment History

### V1: Baseline (NFCorpus)
- **Result**: Agent collapsed to immediate termination (nDCG = 0.0)
- **Issue**: No action masking, agent learned to terminate without searching

### V2: Higher Entropy (NFCorpus)
- **Changes**: Increased entropy coefficient (0.01 → 0.05), added step penalty
- **Result**: Still collapsed to immediate termination
- **Issue**: Reward bug - terminating without search had no penalty

### V3: Action Masking Fix (NFCorpus)
- **Changes**: 
  - Fixed reward bug (nDCG now adds to step penalty, not overwrites)
  - Added action masking (must search before terminate/reformulate)
  - Penalty for terminating without results (-0.5)
- **Result**: Agent now does search → terminate (nDCG ≈ 0.77)
- **Issue**: Still matches single-pass baseline, never uses reformulation

### V4: Curriculum Learning (NFCorpus)
- **Changes**: Focus training on hard queries (recall < 0.6)
- **Result**: Same as V3 - agent learns single-pass is optimal
- **Conclusion**: Reformulation doesn't help on NFCorpus dataset

### V5: FIQA Dataset (Current)
- **Changes**: 
  - New dataset (Financial Q&A - more ambiguous queries)
  - Multi-GPU support (reward model on GPU0, reformulator on GPU1)
- **Status**: Training in progress
- **Hypothesis**: Reformulation should help on harder, more ambiguous queries

---

## Key Files Modified

| File | Changes |
|------|---------|
| `src/config.py` | Added CurriculumConfig, GPU IDs for multi-GPU |
| `src/environment.py` | Action masking, step penalty, has_searched tracking |
| `src/policy.py` | Action masking support in get_action() |
| `src/reward.py` | Multi-GPU support (gpu_id parameter) |
| `src/reformulator.py` | Multi-GPU support (gpu_id parameter) |
| `src/evaluation.py` | Action masking in evaluation loop |
| `train.py` | Curriculum learning, multi-GPU args, action tracking |

---

## Server Setup (Charizard - 2x H100)

### Environment
```bash
cd /mnt/data1/srini/local-m2/search-rl
conda activate searchrl
# or: source .venv/bin/activate
```

### Current Training Command (FIQA)
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataset fiqa --device cuda --episodes 10000 --entropy-coef 0.05 --step-penalty 0.02 --reward-gpu 0 --reformulator-gpu 1 --log-dir runs_fiqa --checkpoint-dir checkpoints_fiqa
```

### Evaluate After Training
```bash
CUDA_VISIBLE_DEVICES=2,3 python eval.py --checkpoint checkpoints_fiqa/policy_final.pt --dataset fiqa --device cuda --num-queries 200
```

### Monitor Training
```bash
tensorboard --logdir runs_fiqa --port 6007
```

---

## Directory Structure

```
search-rl/
├── checkpoints/           # V1 baseline checkpoints
├── checkpoints_v2/        # V2 higher entropy
├── checkpoints_v3/        # V3 action masking
├── checkpoints_v4_curriculum/  # V4 curriculum learning
├── checkpoints_fiqa/      # V5 FIQA dataset (current)
├── runs/                  # TensorBoard logs V1
├── runs_v2/               # TensorBoard logs V2
├── runs_v3/               # TensorBoard logs V3
├── runs_v4_curriculum/    # TensorBoard logs V4
├── runs_fiqa/             # TensorBoard logs V5 (current)
├── data/
│   ├── nfcorpus/          # NFCorpus dataset + index
│   └── fiqa/              # FIQA dataset + index
└── src/                   # Source code
```

---

## Key Results Summary

| Version | Dataset | nDCG@10 | Actions | Notes |
|---------|---------|---------|---------|-------|
| V1 | nfcorpus | 0.0000 | A3:1.00 | Immediate termination |
| V2 | nfcorpus | 0.0000 | A3:1.00 | Still immediate termination |
| V3 | nfcorpus | 0.7708 | A0:0.50 A3:0.50 | Search→Terminate (= single-pass) |
| V4 | nfcorpus | 0.7681 | A0:0.50 A3:0.50 | Same as V3 |
| Baseline | nfcorpus | 0.7724 | A0:0.50 A3:0.50 | Single-pass baseline |
| Heuristic | nfcorpus | 0.7003 | Uses A1,A2 | Reformulation hurts! |

**Key Insight**: On NFCorpus, reformulation (narrow/broaden) actually hurts performance. The RL agent correctly learned not to use it.

---

## Commands Reference

### Pull Latest Code
```bash
cd /mnt/data1/srini/local-m2/search-rl
git pull origin main
```

### Generate Queries for New Dataset
```bash
# Download and prep dataset
cd data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<dataset>.zip
unzip <dataset>.zip
mv <dataset> <dataset>-2018  # If needed for BEIR naming
cd ..

# Generate synthetic queries
CUDA_VISIBLE_DEVICES=2,3 python generate_queries.py --dataset <dataset> --device cuda
```

### Training (Single GPU)
```bash
CUDA_VISIBLE_DEVICES=3 python train.py --dataset nfcorpus --device cuda --episodes 10000 --log-dir runs --checkpoint-dir checkpoints
```

### Training (Multi-GPU - 2x H100)
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataset fiqa --device cuda --episodes 10000 --reward-gpu 0 --reformulator-gpu 1 --log-dir runs_fiqa --checkpoint-dir checkpoints_fiqa
```

### Training with Curriculum Learning
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataset fiqa --device cuda --episodes 10000 --curriculum --hard-threshold 0.6 --hard-ratio 0.7 --log-dir runs_curriculum --checkpoint-dir checkpoints_curriculum
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=2,3 python eval.py --checkpoint <checkpoint_path> --dataset <dataset> --device cuda --num-queries 200
```

### Benchmark Device
```bash
python benchmark.py --device cuda
```

---

## Next Steps (For New Agent)

1. **Wait for FIQA training to complete** (~1-2 hours)
2. **Evaluate FIQA results**:
   ```bash
   CUDA_VISIBLE_DEVICES=2,3 python eval.py --checkpoint checkpoints_fiqa/policy_final.pt --dataset fiqa --device cuda --num-queries 200
   ```
3. **If reformulation still doesn't help**:
   - Try `scifact` or `arguana` datasets
   - Consider improving the reformulator (larger LLM, better prompts)
   - Add intermediate rewards for reformulation quality

4. **If reformulation helps on FIQA**:
   - The agent should learn to use narrow/broaden actions
   - Compare action distributions with NFCorpus results
   - Document findings

---

## Configuration Defaults

```python
# PPO
entropy_coef = 0.05      # Exploration (increased from 0.01)
step_penalty = 0.02      # Per-step cost
gamma = 0.99             # Discount factor
clip_epsilon = 0.2       # PPO clipping

# Environment
max_steps = 5            # Max actions per episode
top_k = 10               # Retrieved documents

# Curriculum (if enabled)
hard_threshold = 0.6     # Recall below this = hard query
hard_ratio = 0.7         # 70% hard queries in training
warmup_episodes = 1000   # Normal sampling before curriculum
```

---

## Troubleshooting

### Corrupted Dataset Download
```bash
rm -f data/<dataset>.zip
# Re-download manually with wget
```

### Multi-line Command Issues
Always use single-line commands or escape properly:
```bash
# Good (single line)
python train.py --dataset fiqa --device cuda --episodes 10000

# Bad (multi-line with issues)
python train.py \
    --dataset fiqa \  # Extra spaces can cause problems
```

### GPU Memory Issues
- Each Qwen-3B model needs ~7GB VRAM
- With 2x H100 (80GB each), there's plenty of headroom
- If OOM, reduce batch size: `--batch-size 32`

---

## Contact

Repository: https://github.com/ch33nchan/search-rl
Branch: main
