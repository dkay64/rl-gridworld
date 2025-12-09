from envs.gridworld import GridWorldEnv
from agents.q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt


def train_q_learning(
    episodes: int = 1000,
    max_steps_per_episode: int = 100,
):
    env = GridWorldEnv(grid_size=4)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
    )

    returns = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        returns.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {ep + 1}/{episodes}, avg return (last 100): {avg_return:.3f}")

    # Plot learning curve
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-learning on GridWorld")
    plt.tight_layout()
    plt.show()

    return agent, returns


if __name__ == "__main__":
    train_q_learning()
