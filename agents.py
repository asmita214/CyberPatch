import numpy as np
import random

# ============================================================
# BASE AGENT — all agents inherit from this
# ============================================================

class BaseAgent:
    def __init__(self, start_node=0):
        self.current_node = start_node
        self.total_reward = 0
        self.moves = 0

    def reset(self, start_node=0):
        self.current_node = start_node
        self.total_reward = 0
        self.moves = 0

    def move(self, action):
        self.current_node = action
        self.moves += 1


# ============================================================
# AGENT 1 — RANDOM (baseline, no intelligence)
# Moves randomly, patches randomly
# Used to show worst case performance
# ============================================================

class RandomAgent(BaseAgent):
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'Random'
        self.agent_type = 'random'

    def select_action(self, env):
        # Just picks a random neighbor — no thinking
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node
        return random.choice(neighbors)


# ============================================================
# AGENT 2 — GREEDY (deterministic baseline)
# Always moves to the highest risk neighbor
# Better than random but never learns or adapts
# ============================================================

class GreedyAgent(BaseAgent):
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'Greedy'
        self.agent_type = 'patcher'

    def select_action(self, env):
        # Always goes to highest risk neighbor
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node
        return max(neighbors, key=lambda n: env.node_risks[n])


# ============================================================
# AGENT 3 — SCANNER BOT (part of our RL team)
# Fast, cheap, cannot patch
# Job: explore network, flag dangerous nodes
# ============================================================

class ScannerBot(BaseAgent):
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'Scanner'
        self.agent_type = 'scanner'

        # Scanner is fast and cheap
        self.speed = 2          # can move 2 nodes per step
        self.energy_cost = 5    # costs 5 energy per move
        self.max_energy = 100   # total energy per episode
        self.energy = self.max_energy

        # Memory: nodes the scanner has visited and flagged
        self.flagged = set()
        self.visited = set()

    def reset(self, start_node=0):
        super().reset(start_node)
        self.energy = self.max_energy
        self.flagged = set()
        self.visited = set()

    def select_action(self, env, brain=None):
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node

        # If RL brain is provided use it
        if brain is not None:
            state = env.get_state()
            return brain.select_action(state, neighbors)

        # Default: prefer unvisited neighbors
        unvisited = [n for n in neighbors if n not in self.visited]
        if unvisited:
            return random.choice(unvisited)
        return random.choice(neighbors)

    def flag_node(self, node, risk):
        # Mark a node as dangerous for the patcher
        if risk >= 2:
            self.flagged.add(node)
        self.visited.add(node)


# ============================================================
# AGENT 4 — PATCHER BOT (part of our RL team)
# Slow, expensive, actually fixes nodes
# Job: follow scanner intel, patch high priority nodes
# ============================================================

class PatcherBot(BaseAgent):
    def __init__(self, start_node=0):
        super().__init__(start_node)
        self.name = 'Patcher'
        self.agent_type = 'patcher'

        # Patcher is slow and expensive
        self.speed = 1          # moves 1 node per step
        self.energy_cost = 15   # costs 15 energy per move
        self.max_energy = 100   # total energy per episode
        self.energy = self.max_energy

        # Patcher tracks what it has already patched
        self.patched = set()

    def reset(self, start_node=0):
        super().reset(start_node)
        self.energy = self.max_energy
        self.patched = set()

    def select_action(self, env, brain=None, flagged_nodes=None):
        neighbors = env.get_possible_actions(self.current_node)
        if not neighbors:
            return self.current_node

        # If RL brain is provided use it
        if brain is not None:
            state = env.get_state()
            return brain.select_action(state, neighbors)

        # Default smart fallback:
        # Priority 1 — flagged high risk neighbors
        if flagged_nodes:
            flagged_neighbors = [n for n in neighbors if n in flagged_nodes]
            if flagged_neighbors:
                return max(flagged_neighbors, key=lambda n: env.node_risks[n])

        # Priority 2 — highest risk neighbor
        return max(neighbors, key=lambda n: env.node_risks[n])

    def record_patch(self, node):
        self.patched.add(node)


# ============================================================
# AGENT FACTORY — easy way to create agents by name
# ============================================================

def create_agent(agent_name, start_node=0):
    agents = {
        'random':  RandomAgent(start_node),
        'greedy':  GreedyAgent(start_node),
        'scanner': ScannerBot(start_node),
        'patcher': PatcherBot(start_node),
    }
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {list(agents.keys())}")
    return agents[agent_name]