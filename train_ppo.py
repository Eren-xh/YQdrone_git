"""Train PPO for the XH1 post-impact recovery task.

Run from the ``drone_xh1`` directory, for example:

    python train_ppo.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .tasks.impact_recovery import ImpactRecoveryEnv
except ImportError:  # Running train_ppo.py directly from the project directory.
    from tasks.impact_recovery import ImpactRecoveryEnv


def make_env():
    return ImpactRecoveryEnv(render_mode=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("runs/xh1_ppo"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(str(args.output))
    vec_env.save(str(args.output) + "_vecnormalize.pkl")
    vec_env.close()


if __name__ == "__main__":
    main()


'''








'''





