import gymnasium as gym
import numpy as np
from collections import deque

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies UDR or ADR to the PandaGym environment.
    """
    def __init__(
        self,
        env,
        mass_range=(0.5, 8.0), # Global range limits (Source is 1.0, Target is 5.0)
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # Absolute global limits beyond which we cannot expand
        self.mass_min_limit, self.mass_max_limit = mass_range

        # Initialize current bounds
        if self.mode == "udr":
            # In UDR, the current range is immediately the maximum global range
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
        else:
            # In ADR (and none), the range starts centered on the Source mass (1.0 kg)
            self.mass_min = 1.0
            self.mass_max = 1.0

        self.last_sample_type = "none"

        # --- ADR specific variables ---
        if self.mode == "adr":
            self.adr_step = 0.25  # How many kg to expand the boundary by each time
            self.performance_buffer = deque(maxlen=20) # History of the last 20 episodes
            self.threshold_high = 0.8 # If success rate >= 80%, expand the range
            self.current_bound_evaluating = "none"

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        elif self.mode == "udr":
            # Sample uniformly between global limits
            self.last_sample_type = "uniform"
            return np.random.uniform(self.mass_min, self.mass_max)

        elif self.mode == "adr":
            # OpenAI's ADR samples uniformly 50% of the time,
            # and 50% at the boundaries to evaluate expansion.
            p = np.random.rand()
            if p < 0.5 or self.mass_min == self.mass_max:
                self.last_sample_type = "uniform"
                self.current_bound_evaluating = "none"
                return np.random.uniform(self.mass_min, self.mass_max)
            elif p < 0.75:
                self.last_sample_type = "lower_bound"
                self.current_bound_evaluating = "lower"
                return self.mass_min
            else:
                self.last_sample_type = "upper_bound"
                self.current_bound_evaluating = "upper"
                return self.mass_max
        else:
            raise NotImplementedError(f"Strategy '{self.mode}' not implemented.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # -----------------------
        # ADR expansion logic
        # -----------------------
        if done and self.mode == "adr":
            # Retrieve whether the arm pushed the cube to the destination (1.0) or not (0.0)
            is_success = info.get("is_success", 0.0)
            self.performance_buffer.append(is_success)

            # If we collected enough data to evaluate the current boundary
            if len(self.performance_buffer) == self.performance_buffer.maxlen:
                success_rate = np.mean(self.performance_buffer)

                # If evaluating upper bound and agent is successful, increase max mass
                if self.current_bound_evaluating == "upper" and success_rate >= self.threshold_high:
                    self.mass_max = min(self.mass_max + self.adr_step, self.mass_max_limit)
                    self.performance_buffer.clear() # Clear the buffer to restart evaluation

                # If evaluating lower bound and agent is successful, decrease min mass
                elif self.current_bound_evaluating == "lower" and success_rate >= self.threshold_high:
                    self.mass_min = max(self.mass_min - self.adr_step, self.mass_min_limit)
                    self.performance_buffer.clear()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):
        # Sample new mass based on UDR or ADR strategy
        new_mass = self._sample_mass()

        if new_mass is not None:
            # Modifies PyBullet physics to inject the new mass into the cube
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            # Useful print to understand what's happening (disable if it floods the console)
            # print(f"[{self.mode}] mass={new_mass:.2f} range=[{self.mass_min:.2f},{self.mass_max:.2f}] type={self.last_sample_type}")

        return super().reset(**kwargs)