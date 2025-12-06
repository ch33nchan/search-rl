# SearchRL

Reinforcement Learning for Agentic Document Retrieval.

> **New to the project?** See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for a comprehensive introduction to the architecture, design decisions, and how everything fits together.

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

##results & updates 
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ tmux capture-pane -t qwen-batch -p | tail -50
model-00002-of-00002.safetensors: 100%|████████████████████| 2.20G/2.20G [00:05<00:00, 388MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████| 3.97G/3.97G [00:06<00:00, 633MB/s]
Fetching 2 files: 100%|██████████████████████████████████████████| 2/2 [00:06<00:00,  3.25s/it]
Loading checkpoint shards: 100%|█████████████████████████████████| 2/2 [00:01<00:00,  1.61it/s]
generation_config.json: 100%|█████████████████████████████████| 242/242 [00:00<00:00, 2.61MB/s]
Generating single-doc queries: 100%|█████████████████████████| 166/166 [01:55<00:00,  1.44it/s]^[[>0;276;0c
Generating multi-doc queries: 100%|█████████████████████████████████████████████| 100/100 [01:34<00:00,  1.05it/s]
Generating adversarial queries: 100%|███████████████████████████████████████████| 100/100 [00:39<00:00,  2.51it/s]
modules.json: 100%|██████████████████████████████████████████████████████████████| 349/349 [00:00<00:00, 2.98MB/s]
config_sentence_transformers.json: 100%|██████████████████████████████████████████| 116/116 [00:00<00:00, 826kB/s]
README.md: 10.5kB [00:00, 25.3MB/s]
sentence_bert_config.json: 100%|████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 376kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████| 612/612 [00:00<00:00, 4.52MB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████| 90.9M/90.9M [00:00<00:00, 109MB/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████| 350/350 [00:00<00:00, 2.49MB/s]
vocab.txt: 232kB [00:00, 15.1MB/s]
tokenizer.json: 466kB [00:00, 11.9MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████| 112/112 [00:00<00:00, 780kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.18MB/s]
Batches: 100%|████████████████████████████████████████████████████████████████████| 57/57 [00:02<00:00, 26.84it/s]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.34it/s]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.63it/s]
Training: 100%|█████████████████████████████████████████████████████████████| 10000/10000 [59:39<00:00,  2.79it/s]
(.venv) srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ 0;276;0c
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ CUDA_VISIBLE_DEVICES=3 uv run python eval.py --checkpoint checkpoints/policy_final.pt --dataset nfcorpus --device cuda --num-queries 200
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3633/3633 [00:00<00:00, 177055.26it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.47it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:0Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.59it/s]
Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  94%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  96%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  96%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  96%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  97%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  98%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating:  99%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:25<00:00,  2.33it/s]
Evaluating single_pass:  10%|████▉                                        Evaluating single_pass:  10%|█████▏                                                                                                          | 20/20                                                                                          Evaluating single_pass:  10%|█████▍                                                                                                                                                                 Evaluating single_pass: 100%|███████████████████████████████████████████████████| 200/200 [01:24<00:00,  2.38it/s]
Evaluating double_reform: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:21<00:00,  1.01s/it]

======================================================================
Method               nDCG@10      Avg Steps    Actions
======================================================================
rl_agent             0.7707       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
single_pass          0.7735       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
heuristic            0.7131       5.00         A0:0.60 A1:0.20 A2:0.20 A3:0.00
======================================================================

Difficulty Breakdown:

rl_agent:
  easy: count=156, nDCG=0.9200 +/- 0.0699
  medium: count=15, nDCG=0.5591 +/- 0.0737
  hard: count=29, nDCG=0.0769 +/- 0.1507

single_pass:
  easy: count=155, nDCG=0.9227 +/- 0.0671
  medium: count=16, nDCG=0.5667 +/- 0.0771
  hard: count=29, nDCG=0.0902 +/- 0.1601

heuristic:
  easy: count=137, nDCG=0.9102 +/- 0.0695
  medium: count=25, nDCG=0.5816 +/- 0.0786
  hard: count=38, nDCG=0.0891 +/- 0.1507
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git add .
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git commit -m "--completed runs"
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'srinivas@charizard.(none)')
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git config user.email "srinivas.tb@dashverse.ai"
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git config user.name  "srinivas"
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git commit -m "--completed runs"
[main 5261f89] --completed runs
 1 file changed, 2945 insertions(+)
 create mode 100644 uv.lock
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git add .
^[[Asrinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git commit -m "--completed runs"
[main 9271479] --completed runs
 11 files changed, 4 insertions(+), 4 deletions(-)
 create mode 100644 checkpoints/policy_1000.pt
 create mode 100644 checkpoints/policy_2000.pt
 create mode 100644 checkpoints/policy_3000.pt
 create mode 100644 checkpoints/policy_4000.pt
 create mode 100644 checkpoints/policy_5000.pt
 create mode 100644 checkpoints/policy_6000.pt
 create mode 100644 checkpoints/policy_7000.pt
 create mode 100644 checkpoints/policy_8000.pt
 create mode 100644 checkpoints/policy_9000.pt
 create mode 100644 checkpoints/policy_final.pt
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ git push origin main
Enumerating objects: 19, done.
Counting objects: 100% (19/19), done.
Delta compression using up to 128 threads
Compressing objects: 100% (17/17), done.
Writing objects: 100% (17/17), 15.11 MiB | 7.59 MiB/s, done.
Total 17 (delta 12), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (12/12), completed with 2 local objects.
To https://github.com/ch33nchan/search-rl
   4515b7c..9271479  main -> main
srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ CUDA_VISIBLE_DEVICES=3 uv run python benchmark.py --device cuda
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead

============================================================
Benchmarking on device: cuda
============================================================

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3633/3633 [00:00<00:00, 181390.69it/s]
1. Retriever initialization...
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.08it/s]
   Time: 1.94s

2. Reward model initialization...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.63it/s]
   Time: 2.18s

3. Reward scoring (10 docs)...
   Time: 0.61s (0.06s per doc)

4. Reformulator initialization...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.58it/s]
   Time: 2.11s

5. Query reformulation...
   Time: 0.31s

6. Policy forward pass...
   Time: 0.06s (0.59ms per forward)

============================================================
Estimated Training Time (10k episodes):
============================================================
Per episode: ~0.75s
10k episodes: ~2.1 hours
With 5 steps/episode avg: ~3.1 hours
============================================================

srinivas@charizard:/mnt/data1/srini/local-m2/search-rl$ CUDA_VISIBLE_DEVICES=3 uv run python eval.py --checkpoint checkpoints/policy_1000.pt --dataset nfcorpus --device cuda --num-queries 200
CUDA_VISIBLE_DEVICES=3 uv run python eval.py --checkpoint checkpoints/policy_5000.pt --dataset nfcorpus --device cuda --num-queries 200
CUDA_VISIBLE_DEVICES=3 uv run python eval.py --checkpoint checkpoints/policy_9000.pt --dataset nfcorpus --device cuda --num-queries 200
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3633/3633 [00:00<00:00, 179904.44it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.48it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.54it/s]
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 120.25it/s]
Evaluating single_pass: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:25<00:00,  2.33it/s]
Evaluating double_reform: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:34<00:00,  1.07s/it]

======================================================================
Method               nDCG@10      Avg Steps    Actions
======================================================================
rl_agent             0.0000       1.00         A0:0.00 A1:0.00 A2:0.00 A3:1.00
single_pass          0.7756       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
heuristic            0.7206       5.00         A0:0.60 A1:0.20 A2:0.20 A3:0.00
======================================================================

Difficulty Breakdown:

rl_agent:
  hard: count=200, nDCG=0.0000 +/- 0.0000

single_pass:
  easy: count=155, nDCG=0.9239 +/- 0.0671
  medium: count=17, nDCG=0.5688 +/- 0.0762
  hard: count=28, nDCG=0.0807 +/- 0.1547

heuristic:
  easy: count=133, nDCG=0.9140 +/- 0.0648
  medium: count=33, nDCG=0.5994 +/- 0.0771
  hard: count=34, nDCG=0.0815 +/- 0.1484
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3633/3633 [00:00<00:00, 180353.73it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.48it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.56it/s]
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:27<00:00,  2.28it/s]
Evaluating single_pass: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:25<00:00,  2.33it/s]
Evaluating double_reform: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:28<00:00,  1.04s/it]

======================================================================
Method               nDCG@10      Avg Steps    Actions
======================================================================
rl_agent             0.7735       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
single_pass          0.7719       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
heuristic            0.7308       5.00         A0:0.60 A1:0.20 A2:0.20 A3:0.00
======================================================================

Difficulty Breakdown:

rl_agent:
  easy: count=156, nDCG=0.9199 +/- 0.0689
  medium: count=15, nDCG=0.5712 +/- 0.0769
  hard: count=29, nDCG=0.0902 +/- 0.1601

single_pass:
  easy: count=155, nDCG=0.9222 +/- 0.0685
  medium: count=16, nDCG=0.5738 +/- 0.0761
  hard: count=29, nDCG=0.0779 +/- 0.1527

heuristic:
  easy: count=141, nDCG=0.9140 +/- 0.0668
  medium: count=23, nDCG=0.5658 +/- 0.0898
  hard: count=36, nDCG=0.1183 +/- 0.1588
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3633/3633 [00:00<00:00, 179671.11it/s]
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.48it/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.51it/s]
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:27<00:00,  2.28it/s]
Evaluating single_pass: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:26<00:00,  2.32it/s]
Evaluating double_reform: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:25<00:00,  1.03s/it]

======================================================================
Method               nDCG@10      Avg Steps    Actions
======================================================================
rl_agent             0.7708       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
single_pass          0.7718       2.00         A0:0.50 A1:0.00 A2:0.00 A3:0.50
heuristic            0.7146       5.00         A0:0.60 A1:0.20 A2:0.20 A3:0.00
======================================================================

Difficulty Breakdown:

rl_agent:
  easy: count=155, nDCG=0.9215 +/- 0.0677
  medium: count=16, nDCG=0.5694 +/- 0.0788
  hard: count=29, nDCG=0.0769 +/- 0.1507

single_pass:
  easy: count=156, nDCG=0.9211 +/- 0.0694
  medium: count=15, nDCG=0.5602 +/- 0.0741
  hard: count=29, nDCG=0.0779 +/- 0.1527

heuristic:
  easy: count=135, nDCG=0.9121 +/- 0.0661
  medium: count=28, nDCG=0.5841 +/- 0.0810
  hard: count=37, nDCG=0.0929 +/- 0.1540

trail 1 : 
Line 24: Training: 100%|█████████████████████████████████████████████████████████████| 10000/10000 [59:39<00:00, 2.79it/s]
All 10,000 episodes finished
Total time: 59 minutes 39 seconds
Average rate: 2.79 episodes/second
Command prompt returned (lines 25-26), indicating the script finished