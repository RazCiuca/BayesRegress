
import pytorchrl
import gym

if __name__ == "__main__":

    env = gym.make("Ant-v4", render_mode="rgb_array")
    observation, info = env.reset()

    for i in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        frame = env.render()

        print(i)
    env.close()


