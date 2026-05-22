
import gymnasium as gym
import panda_gym # type: ignore[import-not-found]
def main():
    render = False

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type="target",
        reward_type="dense",
    )
    
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 5

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()


if __name__ == '__main__':
    main()