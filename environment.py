import numpy as np
import networkx as nx
import random

class CyberEnv:
    def __init__(self, num_nodes=15):
        self.num_nodes = num_nodes
        self.graph = self._build_network()
        self.max_energy = 100

        # --- DYNAMIC THREAT SPREAD (key change) ---
        # Much higher spread probability than before
        # This makes Greedy fail — it can't handle surprises
        # RL learns to anticipate and act before spread happens
        self.spread_prob = {0: 0.0, 1: 0.08, 2: 0.15, 3: 0.25}

        # Node types
        self.node_types = self._assign_node_types()
        self.reset()

    def _build_network(self):
        return nx.barabasi_albert_graph(self.num_nodes, 2, seed=42)

    def _assign_node_types(self):
        return np.random.choice([0, 1, 2], size=self.num_nodes, p=[0.6, 0.3, 0.1])

    def reset(self):
        # Start with fewer high risk nodes
        # Threats GROW during episode due to spread
        # This punishes slow/greedy strategies hard
        self.node_risks = np.random.choice(
            [0, 1, 2, 3],
            size=self.num_nodes,
            p=[0.2, 0.3, 0.3, 0.2]
        )
        self.scanner_energy = self.max_energy
        self.patcher_energy = self.max_energy
        self.flagged_nodes  = set()
        self.steps          = 0
        self.spread_events  = 0  # track how many times threats spread
        return self.get_state()

    def get_state(self):
        return tuple(self.node_risks)

    def get_possible_actions(self, current_node):
        return list(self.graph.neighbors(current_node))

    def _spread_threats(self):
        # Threats spread aggressively every step
        # High risk nodes are contagious to neighbors
        new_risks = self.node_risks.copy()
        for node in range(self.num_nodes):
            risk = self.node_risks[node]
            if risk > 0:
                for neighbor in self.graph.neighbors(node):
                    if self.node_risks[neighbor] < risk:
                        if random.random() < self.spread_prob[risk]:
                            # Neighbor risk increases by 1
                            new_risks[neighbor] = min(
                                self.node_risks[neighbor] + 1, 3
                            )
                            self.spread_events += 1
        self.node_risks = new_risks

    def step(self, current_node, action, agent_type='patcher'):
        reward = 0
        done   = False
        info   = {}
        self.steps += 1

        # Energy cost
        if agent_type == 'scanner':
            energy_cost = 5
            self.scanner_energy -= energy_cost
            if self.scanner_energy <= 0:
                reward -= 5
                done = True
                info['reason'] = 'Scanner out of energy'
                return self.get_state(), reward, done, info
        else:
            energy_cost = 15
            self.patcher_energy -= energy_cost
            if self.patcher_energy <= 0:
                reward -= 5
                done = True
                info['reason'] = 'Patcher out of energy'
                return self.get_state(), reward, done, info

        # Move penalty
        reward -= 1

        if agent_type == 'scanner':
            self.flagged_nodes.add(action)
            risk = self.node_risks[action]
            if risk == 3:
                reward += 3
            elif risk == 2:
                reward += 1

        else:
            risk = self.node_risks[action]

            # Reward by risk level
            if risk == 3:
                reward += 10
            elif risk == 2:
                reward += 5
            elif risk == 1:
                reward += 1
            else:
                reward -= 2  # wasted move on safe node

            # Node type bonus
            node_type = self.node_types[action]
            if risk > 0:
                if node_type == 1:
                    reward += 3
                elif node_type == 2:
                    reward += 5

            # Coordination bonus
            if action in self.flagged_nodes and risk > 0:
                reward += 2
                self.flagged_nodes.discard(action)

            # --- NEW: Urgency bonus ---
            # Extra reward for patching nodes that have
            # many high risk neighbors (stopping spread)
            neighbors     = list(self.graph.neighbors(action))
            risky_neighbors = sum(1 for n in neighbors if self.node_risks[n] >= 2)
            if risky_neighbors >= 2 and risk > 0:
                reward += risky_neighbors * 2  # big bonus for breaking spread chains

            self.node_risks[action] = 0

        # Threats spread every step — this is what kills Greedy
        self._spread_threats()

        # --- NEW: Penalty for total network risk ---
        # Every step the total risk in the network is too high = penalty
        # This teaches RL to act fast and prioritize
        total_risk = np.sum(self.node_risks)
        if total_risk > 20:
            reward -= 3  # network is getting out of control
        elif total_risk > 10:
            reward -= 1

        if np.sum(self.node_risks) == 0:
            done = True
            reward += 20  # big bonus for clearing entire network
            info['reason'] = 'All nodes patched'

        return self.get_state(), reward, done, info

    def get_node_info(self, node):
        type_names = {0: 'endpoint', 1: 'server', 2: 'database'}
        risk_names = {0: 'safe', 1: 'low', 2: 'medium', 3: 'high'}
        return {
            'node':        node,
            'type':        type_names[self.node_types[node]],
            'risk':        risk_names[self.node_risks[node]],
            'connections': len(list(self.graph.neighbors(node))),
            'flagged':     node in self.flagged_nodes
        }

    def get_highest_risk_neighbor(self, current_node):
        neighbors = self.get_possible_actions(current_node)
        if not neighbors:
            return current_node
        return max(neighbors, key=lambda n: self.node_risks[n])