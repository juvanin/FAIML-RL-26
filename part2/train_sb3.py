import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC 
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )

    # added
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["ppo", "sac"],
        help="the algorithm to use (ppo o sac)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    # DONE: add randomization wrapper here (for Task 6: UDR/ADR)
    if args.sampling_strategy in ["udr", "adr"]:
        # Lo implementeremo nella prossima fase!
        env = RandomizationWrapper(env, mode=args.sampling_strategy)
        pass

    # DONE: create model and train it
    print(f"Iniziando l'addestramento con {args.algo.upper()} su ambiente {args.env_type} per {args.timesteps} timesteps...")
    
    # 1. Istanziazione del modello
    if args.algo == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif args.algo == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1)

    # 2. Addestramento (il vero e proprio 'learn')
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # DONE: model.save(save_name)
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    model.save(save_name)
    print(f"Modello salvato con successo: {save_name}.zip")
    env.close()

if __name__ == "__main__":
    main()