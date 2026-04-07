import numpy as np
import random
from collections import defaultdict

class QLearningBrain:
    def __init__(self):
        self.learning_rate = 0.2    # higher = learns faster from dynamic changes
        self.discount      = 0.85   # slightly lower = focus on immediate rewards
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.997

        self.q_table       = defaultdict(lambda: defaultdict(float))
        self.episode       = 0
        self.total_updates = 0

    def simplify_state(self, env, current_node, energy):
        high_risk   = int(np.sum(env.node_risks == 3))
        medium_risk = int(np.sum(env.node_risks == 2))
        node_risk   = int(env.node_risks[current_node])

        if energy > 66:
            energy_bucket = 2
        elif energy > 33:
            energy_bucket = 1
        else:
            energy_bucket = 0

        neighbors = list(env.graph.neighbors(current_node))
        if neighbors:
            neighbor_risks      = [env.node_risks[n] for n in neighbors]
            max_neighbor_risk   = int(max(neighbor_risks))
            high_neighbors      = min(int(sum(1 for r in neighbor_risks if r == 3)), 3)
            avg                 = np.mean(neighbor_risks)
            neighbor_area       = 2 if avg > 2.0 else (1 if avg > 1.0 else 0)
        else:
            max_neighbor_risk = 0
            high_neighbors    = 0
            neighbor_area     = 0

        # Threat pressure — how fast is network degrading?
        threat_pressure = 2 if high_risk >= 5 else (1 if high_risk >= 2 else 0)

        # Total network risk bucket
        total_risk = int(np.sum(env.node_risks))
        risk_bucket = 2 if total_risk > 20 else (1 if total_risk > 10 else 0)

        flagged_nearby = min(
            sum(1 for n in neighbors if n in env.flagged_nodes), 3
        )

        return (
            high_risk,
            medium_risk,
            node_risk,
            energy_bucket,
            max_neighbor_risk,
            neighbor_area,
            high_neighbors,
            threat_pressure,
            risk_bucket,       # NEW: overall network state
            flagged_nearby
        )

    def select_action(self, state, possible_actions, env=None, current_node=None):
        if not possible_actions:
            return None

        if random.random() < self.epsilon:
            # Smart exploration: bias toward high risk neighbors
            if env is not None and current_node is not None and random.random() < 0.7:
                return max(possible_actions, key=lambda n: env.node_risks[n])
            return random.choice(possible_actions)

        q_values  = {a: self.q_table[state][a] for a in possible_actions}
        max_q     = max(q_values.values())
        best_acts = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_acts)

    def update(self, state, action, reward, next_state, next_actions, done):
        current_q = self.q_table[state][action]
        if done or not next_actions:
            target = reward
        else:
            next_qs = [self.q_table[next_state][a] for a in next_actions]
            target  = reward + self.discount * max(next_qs)
        self.q_table[state][action] += self.learning_rate * (target - current_q)
        self.total_updates += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def get_stats(self):
        return {
            'episode':       self.episode,
            'epsilon':       round(self.epsilon, 4),
            'q_table_size':  len(self.q_table),
            'total_updates': self.total_updates
        }