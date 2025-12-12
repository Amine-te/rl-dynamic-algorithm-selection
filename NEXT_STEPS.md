# Next Steps (SB3-Only PPO)

We are switching to Stable-Baselines3 (SB3) PPO and retiring the custom PPO trainer for training/evaluation. Core environment/algorithms remain unchanged.

## Current Status
- ✅ Core env, context manager, reward, state extractor
- ✅ Algorithms: GA, SA, TS (ACO pending)
- ✅ SB3 training script: `experiments/train_sb3.py`
- ✅ SB3 evaluation script: `experiments/evaluate_sb3.py`
- ✅ Gym wrapper: `core/gym_wrapper.py`
- ✅ Generalization fixes in `core/state_extractor.py` (normalized features)
- ✅ Custom PPO code removed (legacy cleaned up)

## Do This Now (SB3)
1) Quick sanity run (small):
```bash
python experiments/train_sb3.py \
  --total_timesteps 100000 \
  --num_cities 20 \
  --max_evaluations 5000 \
  --schedule_interval 500 \
  --checkpoint_dir checkpoints/test_sb3
``` 

2) Evaluate the quick run:
```bash
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/test_sb3 \
  --model_file best_model/best_model.zip \
  --num_test_instances 10
```

3) Main training:
```bash
python experiments/train_sb3.py \
  --total_timesteps 500000 \
  --num_cities 50 \
  --instance_type mixed \
  --max_evaluations 20000 \
  --schedule_interval 1000 \
  --checkpoint_dir checkpoints/tsp50_sb3
```

4) Main evaluation + transfer:
```bash
python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/tsp50_sb3 \
  --model_file best_model/best_model.zip \
  --num_test_instances 20 \
  --compare_baselines

python experiments/evaluate_sb3.py \
  --checkpoint_dir checkpoints/tsp50_sb3 \
  --num_cities 75 \
  --num_test_instances 10
```

## Priorities
- High: Run SB3 training/eval (above). Verify rewards/selection patterns.
- Medium: Add ACO to algorithm pool if desired.
- Medium: Add TensorBoard monitoring (already enabled in SB3 script).
- Medium: Add TSPLIB loader for benchmark evaluation.

## Notes
- Feature dimension depends on number of algorithms; keep algorithm count consistent between train/test.
- All features are normalized for cross-size generalization (20↔100 cities).
- SB3 outputs: `checkpoints/<run>/best_model/`, `checkpoints/<run>/tensorboard/`, `checkpoints/<run>/checkpoints/`.

## If Issues Arise
- Check rewards: use `improvement_with_efficiency` (default).
- Check feature ranges: run `experiments/test_generalization.py` (if present) or inspect state stats during rollout.
- Adjust `schedule_interval` (500, 1000, 2500) for trade-off between switching and progress.

**You're in great shape!** All the hard infrastructure work is done. Now it's time to train and see your RL-DAS system learn to select algorithms dynamically! 🎉