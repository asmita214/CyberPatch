import pandas as pd
import random
import numpy as np
from environment import CyberEnv
from agents import ScannerBot, PatcherBot, RandomAgent, GreedyAgent
from rl_brain import QLearningBrain

EPISODES  = 2000
MAX_STEPS = 50
START_NODE = 0

def run_rl_episode(env, scanner, patcher, brain):
    state = env.reset()
    scanner.reset(START_NODE)
    patcher.reset(START_NODE)
    total_reward = 0

    for step in range(MAX_STEPS):

        # --- SCANNER TURN ---
        scanner_actions = env.get_possible_actions(scanner.current_node)
        if scanner_actions and scanner.energy > 0:
            s_state  = brain.simplify_state(env, scanner.current_node, scanner.energy)
            s_action = brain.select_action(
                s_state, scanner_actions, env=env, current_node=scanner.current_node
            )
            _, reward, done, _ = env.step(
                scanner.current_node, s_action, agent_type='scanner'
            )
            scanner.flag_node(s_action, env.node_risks[s_action])
            scanner.move(s_action)
            scanner.energy -= scanner.energy_cost

            s_next_actions = env.get_possible_actions(scanner.current_node)
            s_next_state   = brain.simplify_state(env, scanner.current_node, scanner.energy)
            brain.update(s_state, s_action, reward, s_next_state, s_next_actions, done)
            total_reward += reward
            if done:
                break

        # --- PATCHER TURN ---
        patcher_actions = env.get_possible_actions(patcher.current_node)
        if patcher_actions and patcher.energy > 0:
            p_state  = brain.simplify_state(env, patcher.current_node, patcher.energy)
            p_action = brain.select_action(
                p_state, patcher_actions, env=env, current_node=patcher.current_node
            )
            _, reward, done, _ = env.step(
                patcher.current_node, p_action, agent_type='patcher'
            )
            patcher.record_patch(p_action)
            patcher.move(p_action)
            patcher.energy -= patcher.energy_cost

            p_next_actions = env.get_possible_actions(patcher.current_node)
            p_next_state   = brain.simplify_state(env, patcher.current_node, patcher.energy)
            brain.update(p_state, p_action, reward, p_next_state, p_next_actions, done)
            total_reward += reward
            if done:
                break

    return total_reward


def run_random_episode(env, agent):
    env.reset()
    agent.reset(START_NODE)
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(env)
        _, reward, done, _ = env.step(agent.current_node, action, agent_type='random')
        agent.move(action)
        total_reward += reward
        if done:
            break
    return total_reward


def run_greedy_episode(env, agent):
    env.reset()
    agent.reset(START_NODE)
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = agent.select_action(env)
        _, reward, done, _ = env.step(agent.current_node, action, agent_type='patcher')
        agent.move(action)
        total_reward += reward
        if done:
            break
    return total_reward


def train():
    print("Starting CyberPatch Training (Improved)...")
    print(f"Episodes: {EPISODES} | Max steps: {MAX_STEPS}")
    print("-" * 55)

    env     = CyberEnv(num_nodes=15)
    scanner = ScannerBot(START_NODE)
    patcher = PatcherBot(START_NODE)
    brain   = QLearningBrain()

    random_agent = RandomAgent(START_NODE)
    greedy_agent = GreedyAgent(START_NODE)

    results = []

    for episode in range(1, EPISODES + 1):
        np.random.seed(episode)
        random.seed(episode)

        rl_reward     = run_rl_episode(env, scanner, patcher, brain)
        random_reward = run_random_episode(env, random_agent)
        greedy_reward = run_greedy_episode(env, greedy_agent)

        brain.decay_epsilon()

        results.append({
            'Episode':      episode,
            'RL_Team':      rl_reward,
            'Random_Agent': random_reward,
            'Greedy_Agent': greedy_reward,
            'Q_States':     len(brain.q_table)
            
        })

        if episode % 200 == 0:
            stats   = brain.get_stats()
            avg_rl  = np.mean([r['RL_Team']      for r in results[-200:]])
            avg_rnd = np.mean([r['Random_Agent']  for r in results[-200:]])
            avg_grd = np.mean([r['Greedy_Agent']  for r in results[-200:]])
            print(f"Ep {episode:4d} | "
                  f"RL: {avg_rl:5.1f} | "
                  f"Random: {avg_rnd:5.1f} | "
                  f"Greedy: {avg_grd:5.1f} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Q-states: {stats['q_table_size']}")

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)

    print("-" * 55)
    print("Training complete.")
    last = df.tail(200)
    print(f"\nFinal 200 episodes average:")
    print(f"  RL Team:      {last['RL_Team'].mean():.1f}")
    print(f"  Random Agent: {last['Random_Agent'].mean():.1f}")
    print(f"  Greedy Agent: {last['Greedy_Agent'].mean():.1f}")

    return df

if __name__ == "__main__":
    train()