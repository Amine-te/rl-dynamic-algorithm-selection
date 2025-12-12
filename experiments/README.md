# RL-DAS Training and Evaluation (Stable-Baselines3)

Use Stable-Baselines3 PPO (SB3) instead of the custom PPO trainer.

---

## Training (SB3 PPO)

**Quick start (default):**
```bash
python experiments/train_sb3.py
```

**Custom training (50 cities):**
```bash
python experiments/train_sb3.py \
  --total_timesteps 500000 \
  --num_cities 50 \
  --instance_type mixed \
  --max_evaluations 20000 \
  --schedule_interval 1000 \
  --checkpoint_dir checkpoints/tsp50_sb3
```

**Smaller test run:**
```bash
python experiments/train_sb3.py \
  --total_timesteps 100000 \
  --num_cities 20 \
  --max_evaluations 5000 \
  --schedule_interval 500 \
  --checkpoint_dir checkpoints/test_sb3
```

---

## Evaluation (SB3 PPO)

**Evaluate best model with baselines (compares both cost and execution time):**
```bash
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/tsp50_sb3 \
  --model_file best_model/best_model.zip \
  --num_test_instances 20 \
  --compare_baselines
```

**Transfer to different city sizes:**
```bash
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/tsp50_sb3 \
  --num_cities 75 \
  --num_test_instances 10
```

**Single-line minimal:**
```bash
python experiments/train_sb3.py --total_timesteps 100000 --num_cities 20 --checkpoint_dir checkpoints/test_sb3
python experiments/evaluate_sb3.py --checkpoint_dir checkpoints/test_sb3 --model_file best_model/best_model.zip --num_test_instances 5
```

**Evaluation output includes:**
- RL-DAS vs individual algorithms cost comparison
- RL-DAS vs individual algorithms execution time comparison
- Statistical summary (mean ± std, min/max, improvement percentages)
- Results saved to `evaluation_results_sb3.json`

---