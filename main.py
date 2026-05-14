import gymnasium as gym

def main():
    # Initialize the LunarLander environment
    # render_mode="human" tells the environment to display a window
    env = gym.make("LunarLander-v3", render_mode="human")

    # Reset the environment to get the initial observation
    # We set a seed for reproducibility
    observation, info = env.reset(seed=42)

    # Run for 1000 steps
    for step in range(1000):
        # Sample a random action from the action space
        # Actions: 0=Do nothing, 1=Fire left engine, 2=Fire main engine, 3=Fire right engine
        action = env.action_space.sample()

        # Apply the action to the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # If the lander crashes (terminated) or runs out of time (truncated), reset
        if terminated or truncated:
            print(f"Episode ended at step {step}. Resetting...")
            observation, info = env.reset()

    # Close the environment and the rendering window
    env.close()

if __name__ == "__main__":
    main()