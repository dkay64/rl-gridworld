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

    # Plot learned state values as a heatmap
    plot_state_values(agent, grid_size=4)

    return agent, returns


def run_greedy_episode(env: GridWorldEnv, agent: QLearningAgent, max_steps: int = 50):
    state = env.reset()
    trajectory = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        action = int(np.argmax(agent.Q[state]))
        next_state, reward, done, _ = env.step(action)
        trajectory.append(next_state)
        total_reward += reward
        state = next_state
        if done:
            break

    return trajectory, total_reward


def plot_state_values(agent: QLearningAgent, grid_size: int = 4):
    V = np.max(agent.Q, axis=1)  # value of state = max_a Q(s,a)
    V_grid = V.reshape((grid_size, grid_size))

    plt.figure()
    plt.imshow(V_grid, interpolation="nearest")
    plt.colorbar(label="State value")
    plt.title("Learned state values")
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{V_grid[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Train the agent
    agent, returns = train_q_learning()

    # Evaluate greedily after training
    env = GridWorldEnv(grid_size=4)
    traj, total_reward = run_greedy_episode(env, agent)
    print("Greedy trajectory:", traj)
    print("Total reward:", total_reward)
