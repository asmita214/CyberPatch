import numpy as np
import pandas as pd
import random
import os
import networkx as nx
from environment import CyberEnv
from rl_brain import QLearningBrain
from agents import ScannerBot, PatcherBot, BaseAgent

# ============================================================
# NEW RED TEAM AGENTS
# ============================================================

class RedSpreaderAgent(BaseAgent):
    """
    Moves fast (speed 2x), flags high-connectivity nodes for RedInfector.
    Moves: 20 per episode (Energy 100, Cost 5).
    """
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'RedSpreader'
        self.speed = 2
        self.energy_cost = 5
        self.energy = 100
        self.flagged_hubs = set()

    def reset(self, start_node=0):
        super().reset(start_node)
        self.energy = 100
        self.flagged_hubs = set()

    def select_action(self, env, brain, hubs):
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node
        
        state = brain.simplify_state(env, self.current_node, self.energy)
        action = brain.select_action(state, neighbors, env, self.current_node)
        return action

class RedInfectorAgent(BaseAgent):
    """
    Moves slow (speed 1x), actively increases risk level by +1.
    Moves: 6 per episode (Energy 100, Cost 15).
    """
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'RedInfector'
        self.speed = 1
        self.energy_cost = 15
        self.energy = 100

    def reset(self, start_node=0):
        super().reset(start_node)
        self.energy = 100

    def select_action(self, env, brain):
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node
        
        state = brain.simplify_state(env, self.current_node, self.energy)
        action = brain.select_action(state, neighbors, env, self.current_node)
        return action

# ============================================================
# TRAINING UTILITIES
# ============================================================

def get_hub_nodes(graph, threshold=3):
    return [node for node, degree in dict(graph.degree()).items() if degree > threshold]

def run_adversarial_training(episodes=1000):
    env = CyberEnv(num_nodes=15)
    hubs = get_hub_nodes(env.graph)
    
    blue_brain = QLearningBrain()
    red_brain = QLearningBrain()
    
    # Blue Agents
    scanner = ScannerBot(start_node=0)
    patcher = PatcherBot(start_node=0)
    
    # Red Agents
    spreader = RedSpreaderAgent(start_node=14)
    infector = RedInfectorAgent(start_node=14)
    
    results = []

    print(f"Starting Red vs Blue training for {episodes} episodes...")

    for ep in range(episodes):
        env.reset()
        scanner.reset(0)
        patcher.reset(0)
        spreader.reset(14)
        infector.reset(14)
        
        blue_total_reward = 0
        red_total_reward = 0
        blue_patched_count = 0
        red_infected_count = 0
        
        red_flagged_hubs = set()
        
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            step_count += 1
            
            # --- BLUE TEAM TURN ---
            # Scanner Move
            if scanner.energy > 0:
                s_state = blue_brain.simplify_state(env, scanner.current_node, scanner.energy)
                s_action = scanner.select_action(env, blue_brain)
                _, s_reward, s_done, _ = env.step(scanner.current_node, s_action, 'scanner')
                
                # Update brain
                s_next_state = blue_brain.simplify_state(env, s_action, scanner.energy - scanner.energy_cost)
                s_next_actions = env.get_possible_actions(s_action)
                blue_brain.update(s_state, s_action, s_reward, s_next_state, s_next_actions, s_done)
                
                scanner.move(s_action)
                scanner.energy -= scanner.energy_cost
                blue_total_reward += s_reward
                if s_done: done = True

            # Patcher Move
            if patcher.energy > 0 and not done:
                p_state = blue_brain.simplify_state(env, patcher.current_node, patcher.energy)
                p_action = patcher.select_action(env, blue_brain, env.flagged_nodes)
                
                # Check if it's a patch (risk > 0)
                risk_before = env.node_risks[p_action]
                _, p_reward, p_done, _ = env.step(patcher.current_node, p_action, 'patcher')
                
                if risk_before > 0:
                    blue_patched_count += 1
                
                # Update brain
                p_next_state = blue_brain.simplify_state(env, p_action, patcher.energy - patcher.energy_cost)
                p_next_actions = env.get_possible_actions(p_action)
                blue_brain.update(p_state, p_action, p_reward, p_next_state, p_next_actions, p_done)
                
                patcher.move(p_action)
                patcher.energy -= patcher.energy_cost
                blue_total_reward += p_reward
                if p_done: done = True

            # --- RED TEAM TURN ---
            # Red Spreader Move
            if spreader.energy > 0 and not done:
                rs_state = red_brain.simplify_state(env, spreader.current_node, spreader.energy)
                rs_action = spreader.select_action(env, red_brain, hubs)
                
                # Spreader logic: flags hubs
                rs_reward = -1 # move penalty
                if rs_action in hubs:
                    red_flagged_hubs.add(rs_action)
                    # Reward for flagging hubs? Let's say small reward
                    rs_reward += 2
                
                # Update brain
                rs_next_state = red_brain.simplify_state(env, rs_action, spreader.energy - spreader.energy_cost)
                rs_next_actions = env.get_possible_actions(rs_action)
                red_brain.update(rs_state, rs_action, rs_reward, rs_next_state, rs_next_actions, False)
                
                spreader.move(rs_action)
                spreader.energy -= spreader.energy_cost
                red_total_reward += rs_reward

            # Red Infector Move
            if infector.energy > 0 and not done:
                ri_state = red_brain.simplify_state(env, infector.current_node, infector.energy)
                ri_action = infector.select_action(env, red_brain)
                
                # Infect logic
                risk_before = env.node_risks[ri_action]
                ri_reward = -1 # move penalty
                
                if risk_before < 3:
                    env.node_risks[ri_action] += 1
                    red_infected_count += 1
                    
                    # Reward based on location
                    if ri_action in hubs:
                        ri_reward += 10
                    elif env.node_types[ri_action] == 1: # Medium
                        ri_reward += 5
                    else: # Low
                        ri_reward += 1
                    
                    # Bonus for coordinating with spreader
                    if ri_action in red_flagged_hubs:
                        ri_reward += 5
                        red_flagged_hubs.discard(ri_action)
                
                # Update brain
                ri_next_state = red_brain.simplify_state(env, ri_action, infector.energy - infector.energy_cost)
                ri_next_actions = env.get_possible_actions(ri_action)
                red_brain.update(ri_state, ri_action, ri_reward, ri_next_state, ri_next_actions, False)
                
                infector.move(ri_action)
                infector.energy -= infector.energy_cost
                red_total_reward += ri_reward

            # Check if everyone is out of energy
            if scanner.energy <= 0 and patcher.energy <= 0 and spreader.energy <= 0 and infector.energy <= 0:
                done = True
                
        # End of episode
        blue_brain.decay_epsilon()
        red_brain.decay_epsilon()
        
        results.append({
            'Episode': ep + 1,
            'Blue_Score': blue_total_reward,
            'Red_Score': red_total_reward,
            'Blue_Patched': blue_patched_count,
            'Red_Infected': red_infected_count
        })
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1} complete. Blue Avg: {np.mean([r['Blue_Score'] for r in results[-100:]]):.1f}, Red Avg: {np.mean([r['Red_Score'] for r in results[-100:]]):.1f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('red_blue_results.csv', index=False)
    print("Training finished. Results saved to red_blue_results.csv")

if __name__ == "__main__":
    run_adversarial_training(1000)
