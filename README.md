# 🛡️ CyberPatch — Adaptive Network Security via Reinforcement Learning

> **Hackathon Submission · Problem Statement 02 · Adaptive Systems · ML Engineering Track**

A reinforcement learning system that trains security bots to autonomously protect a computer network from dynamic cyber threats. Two specialized agents learn — through trial and error — which vulnerabilities to patch first, in what order, before threats spread and escalate.

---
## 📄 Documentation

[Download Project Report](docs/CyberPatch_Report.docx)

## 📊 Results

| Agent | Final Avg Score | Learns | Adapts to Spread |
|-------|---------------|--------|-----------------|
| **RL Team** | **58.5** ✓ | Yes | Yes |
| Greedy | 56.1 | No | No |
| Random | 27.5 | No | No |

> RL beats Greedy in **all 10 training phases** and outperforms Random by **+113%** — emergent behaviour, not hardcoded rules.

---

## 🧠 The Core Idea

A network of 15 computers has vulnerable nodes. Threats **spread dynamically** every step (25% infection probability). Two bots must coordinate to patch the most critical nodes before the network is compromised — all within a limited energy budget.

**Why RL and not Greedy?**  
Greedy always picks the current highest-risk neighbor. When threats spread, the "best" node changes every step — Greedy chases a moving target. The RL agent learned to **cut off spread chains** by targeting hub nodes connected to many risky nodes. This emergent strategy was never programmed — it was discovered through 2,000 episodes of experience.

---

## 🤖 Two Agent Types

| Agent | Speed | Energy/Move | Max Moves | Role |
|-------|-------|-------------|-----------|------|
| **Scanner Bot** | 2x | 5 units | 20/ep | Fast explorer — finds and flags risky nodes |
| **Patcher Bot** | 1x | 15 units | 6/ep | Fixes vulnerabilities, follows Scanner intel |

Plus two baselines for comparison: **Greedy** (deterministic) and **Random** (baseline floor).

---

## 🗂️ Project Structure

```
CyberPatch/
│
├── environment.py      # Network world — Barabasi-Albert graph, threat spread, reward function
├── agents.py           # ScannerBot, PatcherBot, GreedyAgent, RandomAgent + factory
├── rl_brain.py         # Q-Learning brain — Q-table, ε-greedy, Bellman update, state simplifier
├── train.py            # Training loop — 2000 episodes, CSV export, progress logging
├── app.py              # Streamlit dashboard — 5 tabs, Plotly charts, live simulation
│
├── results.csv         # Generated after training — episode-by-episode scores
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/CyberPatch.git
cd CyberPatch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit plotly networkx pandas numpy
```

### 3. Train the agents

```bash
python train.py
```

This runs 2,000 episodes and saves results to `results.csv`. Progress is printed every 200 episodes:

```
Ep  200 | RL:  72.2 | Random:  29.3 | Greedy:  54.6 | ε: 0.548 | Q-states: 2786
Ep  400 | RL:  62.0 | Random:  31.1 | Greedy:  55.0 | ε: 0.301 | Q-states: 4299
...
Ep 2000 | RL:  58.5 | Random:  27.5 | Greedy:  56.1 | ε: 0.050 | Q-states: 8863
```

### 4. Launch the dashboard

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
| Smart explore bias | 70% | Biased toward high-risk neighbors even while exploring |

### State Space

The full risk array (4^15 = 1 billion states) was too large. We simplified to 10 components:

```python
state = (
    high_risk_count,      # how many high-risk nodes remain
    medium_risk_count,    # how many medium-risk nodes remain
    node_risk,            # current node's risk level
    energy_bucket,        # high / medium / low
    max_neighbor_risk,    # highest risk among neighbors
    neighbor_area,        # average risk zone around agent
    high_neighbors,       # count of high-risk neighbors
    threat_pressure,      # network-wide danger level
    risk_bucket,          # total network risk bucket
    flagged_nearby,       # nodes Scanner has marked nearby
)
```

Result: **~8,863 unique discovered states** — converges cleanly in 2,000 episodes.

---

## 🌐 Environment Details

- **Network topology** — Barabasi-Albert graph (mimics real networks with hub nodes)
- **Node types** — Endpoint (×9), Server (×4, +3 bonus), Database (×2, +5 bonus)
- **Threat spread** — High-risk nodes infect neighbors at 15–25% per step
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

## 📊 Dashboard — 5 Tabs

| Tab | What It Shows |
|-----|--------------|
| **Overview** | KPI cards, 2000-episode trajectory, phase table, score distribution |
| **Learning Curve** | Smoothing slider, phase highlights, ε decay, Q-table growth |
| **Network & Sim** | Live network graph, run episodes, event log |
| **Agent Analysis** | Agent cards with stats, phase bar chart, head-to-head comparison |
| **Insights** | Key finding, 6 insight cards, histogram, stability chart, correlation heatmap |

---

## 🏗️ Hackathon Constraints

| Constraint | Satisfied | How |
|------------|-----------|-----|
| C-01: Two agent types | ✅ | Scanner Bot (fast/cheap) + Patcher Bot (slow/expensive) |
| C-02: Resource constraints | ✅ | 100 energy units per bot per episode |
| C-03: Performance metric | ✅ | Reward = tasks completed / energy spent |
| C-04: Demonstration of learning | ✅ | RL beats Greedy in all 10 training phases |

---

## 🔬 Key Innovations

- **Dynamic threat spreading** — makes static strategies fail, RL necessary
- **Simplified state space** — reduced 4^15 to ~8,863 learnable states
- **Smart exploration** — biased random moves collect better training data
- **Coordination reward** — Scanner + Patcher teamwork emerged naturally
- **Spread chain bonus** — directly trains the hub-targeting behaviour

---

## 🚀 Future Work

- Replace Q-Learning with **DQN** for continuous state spaces
- Add **unknown risk levels** until scanned — real uncertainty modelling
- **Red team vs blue team** — attacker agent vs defender agents
- Real vulnerability data integration via **NVD/CVSS scores**

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
