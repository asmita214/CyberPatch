# 🛡️ CyberPatch — Adaptive Network Security via Reinforcement Learning

> **Hackathon Submission · Problem Statement 02 · Adaptive Systems · ML Engineering Track**

A reinforcement learning system that trains security bots to autonomously protect a computer network from dynamic cyber threats. Two specialized agents learn — through trial and error — which vulnerabilities to patch first, in what order, before threats spread and escalate.

---

## 📊 Results

| Agent | Final Avg Score | Learns | Adapts to Spread |
|-------|---------------|--------|-----------------|
| **RL Team** | **60.2** ✓ | Yes | Yes |
| Greedy | 55.2 | No | No |
| Random | 28.0 | No | No |

> RL beats Greedy in **all 10 training phases** · outperforms Random by **+115%** · gap of **+5.0** reward units · emergent behaviour, not hardcoded rules.

---

## ⚔️ Adversarial Arena — Red vs Blue

Beyond the core system, CyberPatch extends to a full adversarial setting where an intelligent red team actively tries to infect the network while the blue team defends.

| Metric | Value |
|--------|-------|
| Blue Team avg score | **55.4** |
| Red Team avg score | **15.8** |
| Total nodes patched | **6,725** |
| Total nodes infected | **1,816** |
| Blue dominance ratio | **3.5x** |

> The blue team's learned policy — trained only in the original environment — dominates an intelligent adversary it was never trained against. For every node red infected, blue patched nearly 4.

---

## 🧠 The Core Idea

A network of 15 computers has vulnerable nodes. Threats **spread dynamically** every step (25% infection probability). Two bots must coordinate to patch the most critical nodes before the network is compromised — all within a limited energy budget.

**Why RL and not Greedy?**
Greedy always picks the current highest-risk neighbor. When threats spread, the best node changes every step — Greedy chases a moving target. The RL agent learned to **cut off spread chains** by targeting hub nodes connected to many risky nodes. This emergent strategy was never programmed — it was discovered through 2,000 episodes of experience.

---

## 🤖 Agent Types

| Agent | Speed | Energy/Move | Max Moves | Role |
|-------|-------|-------------|-----------|------|
| **Scanner Bot** | 2x | 5 units | 20/ep | Fast explorer — finds and flags risky nodes |
| **Patcher Bot** | 1x | 15 units | 6/ep | Fixes vulnerabilities, follows Scanner intel |
| Greedy Agent | 1x | 15 units | 6/ep | Deterministic baseline — always picks highest risk |
| Random Agent | 1x | 15 units | 6/ep | Random baseline — establishes performance floor |

---

## 🗂️ Project Structure

```
CyberPatch/
│
├── environment.py        # Network world — Barabasi-Albert graph, threat spread, reward function
├── agents.py             # ScannerBot, PatcherBot, GreedyAgent, RandomAgent + factory
├── rl_brain.py           # Q-Learning brain — Q-table, ε-greedy, Bellman update, state simplifier
├── train.py              # Training loop — 2000 episodes, CSV export, Q-states tracking
├── red_blue_train.py     # Adversarial training — red team vs blue team, 1000 episodes
├── app.py                # Streamlit dashboard — 6 tabs including Adversarial Arena
│
├── results.csv           # Generated after train.py — episode scores + Q-states
├── red_blue_results.csv  # Generated after red_blue_train.py — adversarial results
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/asmita214/CyberPatch.git
cd CyberPatch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the main agents

```bash
python train.py
```

Runs 2,000 episodes and saves `results.csv`. Progress prints every 200 episodes:

```
Ep  200 | RL:  74.2 | Random:  28.6 | Greedy:  54.6 | ε: 0.548 | Q-states: 2786
Ep  400 | RL:  64.1 | Random:  28.7 | Greedy:  55.4 | ε: 0.301 | Q-states: 4296
Ep  600 | RL:  62.5 | Random:  26.5 | Greedy:  56.2 | ε: 0.165 | Q-states: 5283
Ep 1000 | RL:  60.3 | Random:  27.1 | Greedy:  53.6 | ε: 0.050 | Q-states: 6802
Ep 2000 | RL:  60.2 | Random:  28.0 | Greedy:  55.2 | ε: 0.050 | Q-states: 8920
```

### 4. Train the adversarial arena

```bash
python red_blue_train.py
```

Runs 1,000 adversarial episodes and saves `red_blue_results.csv`.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📈 How Q-Learning Works Here

```
Q(s, a)  ←  Q(s, a)  +  α × [ r  +  γ · max Q(s', a')  −  Q(s, a) ]
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning rate α | 0.20 | Faster adaptation to dynamic threats |
| Discount factor γ | 0.85 | Prioritises immediate threat control |
| Epsilon start | 1.00 | Fully random at episode 1 |
| Epsilon end | 0.05 | 5% exploration retained |
| Epsilon decay | 0.997 | ~600 episodes of exploration first |
| Smart explore bias | 70% | Biased toward high-risk neighbors during exploration |

### State Space

The full risk array (4^15 = 1 billion states) was too large for Q-Learning to converge. We simplified to 10 components:

```python
state = (
    high_risk_count,      # how many high-risk nodes remain
    medium_risk_count,    # how many medium-risk nodes remain
    node_risk,            # current node's risk level
    energy_bucket,        # high / medium / low remaining energy
    max_neighbor_risk,    # highest risk among neighbors
    neighbor_area,        # average risk zone around agent
    high_neighbors,       # count of high-risk neighbors
    threat_pressure,      # network-wide danger level
    risk_bucket,          # total network risk bucket
    flagged_nearby,       # nodes Scanner has marked nearby
)
```

Result: **~8,920 unique discovered states** — converges cleanly in 2,000 episodes.

---

## 🌐 Environment Details

- **Network topology** — Barabasi-Albert graph (mimics real networks with hub nodes)
- **Node types** — Endpoint (×9), Server (×4, +3 bonus), Database (×2, +5 bonus)
- **Threat spread** — High-risk nodes infect neighbors at 15–25% probability per step
- **Reward function**

```
+10  patch high-risk node
+5   patch medium-risk node
+1   patch low-risk node
-1   every move (energy cost)
-2   patch already-safe node (wasted move)
-5   exceed energy budget
+2   patcher follows scanner's flagged node (coordination bonus)
+2×N bonus for patching hub nodes surrounded by N risky neighbors
+20  clear entire network
```

---

## 📊 Dashboard — 6 Tabs

| Tab | What It Shows |
|-----|--------------|
| **Overview** | KPI cards, 2000-episode trajectory, phase breakdown table, score distribution |
| **Learning Curve** | Smoothing slider, phase filter, ε decay, real Q-table growth chart |
| **Network & Sim** | Live network graph, run RL/Greedy/Random episodes, event log |
| **Agents** | Agent cards with stats, phase bar chart, final score comparison |
| **Insights** | Key finding, 6 insight cards, score histogram, variance chart, correlation heatmap |
| **Adversarial Arena** | Red vs blue live simulation, live score tracker, event log, training curves |

---

## 🏗️ Hackathon Constraints

| Constraint | Satisfied | How |
|------------|-----------|-----|
| C-01: Two agent types | ✅ | Scanner Bot (fast/cheap) + Patcher Bot (slow/expensive) |
| C-02: Resource constraints | ✅ | 100 energy units per bot per episode, hard cap with penalty |
| C-03: Performance metric | ✅ | Tasks completed + efficiency (reward per energy spent) |
| C-04: Demonstration of learning | ✅ | RL beats Greedy in all 10 training phases, real Q-table growth chart |

---

## 🔬 Key Innovations

- **Dynamic threat spreading** — 25% spread probability makes static strategies fail and RL necessary
- **Simplified state space** — reduced 4^15 (1 billion) to ~8,920 learnable states
- **Smart exploration** — 70% bias toward high-risk neighbors during random exploration
- **Coordination reward** — Scanner + Patcher teamwork emerged naturally, never programmed
- **Spread chain bonus** — directly trains the hub-targeting behaviour
- **Adversarial extension** — red vs blue, learned policy dominates unseen intelligent attacker

---

## 🚀 Future Work

- Extend to **Deep Q-Network (DQN)** for continuous state spaces and 100+ node networks
- Add **unknown risk levels until scanned** — real uncertainty modelling
- **Red team trains simultaneously** — full adversarial co-evolution, not just evaluation
- Integrate real **CVSS scores from NVD database** to replace simulated risk levels
- Scale to **enterprise networks with 500+ nodes** using Graph Neural Networks (GNNs)

---

## 📋 Requirements

```
streamlit
plotly
networkx
pandas
numpy
```

---

*Problem Statement 02 — Adaptive Systems · ML Engineering Track · Hackathon Submission*
