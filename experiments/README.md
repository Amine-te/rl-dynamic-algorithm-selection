# RL-DAS Training and Evaluation (Minimal Version)

This is the simplified version containing **only the universal single-line commands** for training and evaluation.

---

## Training

**Default training:**

```bash
python experiments/train.py
```

**Custom training (50 cities):**

```bash
python experiments/train.py --num_epochs 200 --instances_per_epoch 16 --num_cities 50 --max_evaluations 20000 --schedule_interval 1000 --checkpoint_dir checkpoints/tsp50
```

**Quick test run:**

```bash
python experiments/train.py --num_epochs 50 --instances_per_epoch 8 --num_cities 20 --max_evaluations 5000 --schedule_interval 500 --checkpoint_dir checkpoints/test
```

---

## Evaluation

**Evaluate the best model with baseline comparison:**

```bash
python experiments/evaluate.py --checkpoint_dir checkpoints/tsp50 --checkpoint_file best_model.pt --num_test_instances 20 --compare_baselines
```

**Transfer evaluation on different city size:**

```bash
python experiments/evaluate.py --checkpoint_dir checkpoints/tsp50 --num_cities 75 --num_test_instances 10
```

---