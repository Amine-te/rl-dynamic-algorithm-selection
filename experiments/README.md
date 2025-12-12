# RL-DAS Training and Evaluation (Stable-Baselines3)

Use Stable-Baselines3 PPO (SB3) instead of the custom PPO trainer.

---

## 🚀 Automatic Experiment Management

Experiments are now **automatically organized** with unique names and directories:

- **Auto-naming**: `tsp{CITIES}_{TIMESTEPS}k_{TIMESTAMP}`
- **Auto-directories**: `checkpoints/tsp50_500k_20241201_143022/`
- **No overwrites**: Each run creates a unique experiment

**Example outputs:**
```
checkpoints/
├── tsp20_100k_20241201_143022/    # First run
│   ├── config.json
│   ├── best_model/
│   ├── checkpoints/
│   ├── tensorboard/
│   └── evaluation_results_sb3.json
└── tsp50_500k_20241201_150045/    # Second run
    ├── config.json
    ├── best_model/
    └── ...
```

**List experiments:**
```bash
python experiments/list_experiments.py
```

---

## Training (SB3 PPO)

**Quick start (auto-named):**
```bash
python experiments/train_sb3.py
```

**Custom training:**
```bash
# Auto-named experiment
python experiments/train_sb3.py \
  --total_timesteps 500000 \
  --num_cities 50

# Custom name
python experiments/train_sb3.py \
  --experiment_name my_custom_experiment \
  --total_timesteps 500000 \
  --num_cities 50
```

**Small test run:**
```bash
python experiments/train_sb3.py \
  --total_timesteps 100000 \
  --num_cities 20 \
  --max_evaluations 5000
```

---

## Evaluation (SB3 PPO)

**Find experiment directory:**
```bash
python experiments/list_experiments.py
```

**Evaluate best model with baselines (compares both cost and execution time):**
```bash
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/YOUR_EXPERIMENT_NAME \
  --model_file best_model/best_model.zip \
  --num_test_instances 20 \
  --compare_baselines
```

**Transfer evaluation (test generalization):**
```bash
# Train on 50 cities, test on 75 cities
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/tsp50_500k_20241201_143022 \
  --num_cities 75 \
  --num_test_instances 10
```

**Quick evaluation:**
```bash
# Train and evaluate in sequence
python experiments/train_sb3.py --total_timesteps 100000 --num_cities 20
python experiments/list_experiments.py  # Find the experiment name
python experiments/evaluate_sb3.py --checkpoint_dir checkpoints/LATEST_EXPERIMENT --num_test_instances 5
```

**Evaluation output includes:**
- RL-DAS vs individual algorithms cost comparison
- RL-DAS vs individual algorithms execution time comparison
- Statistical summary (mean ± std, min/max, improvement percentages)
- Results saved to `evaluation_results_sb3.json`

---
---