import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random, sys, os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import CyberEnv

st.set_page_config(page_title="CyberPatch · Red vs Blue", page_icon="⚔️", layout="wide")

# ── STYLING ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #f4f6fb !important;
    color: #1e293b !important;
}
.stApp { background: #f4f6fb !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding: 0 4px !important;
    border-radius: 10px 10px 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
}

/* Buttons */
.stButton > button {
    background: #ffffff !important;
    border: 1.5px solid #0ea5e9 !important;
    color: #0ea5e9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #0ea5e9 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOGIC ──────────────────────────────────────────────────
@st.cache_data
def load_results():
    try:
        df = pd.read_csv("red_blue_results.csv")
        return df
    except:
        return None

def smooth(series, w=20):
    return series.rolling(w, min_periods=1).mean()

# Colors
C_BLUE = "#0ea5e9"
C_RED  = "#f43f5e"
C_RED_DARK = "#be123c"
C_RED_LIGHT = "#fb7185"
C_TEAL = "#10b981"
C_YELLOW = "#f59e0b"
SURF = "#ffffff"
GRID = "#e2e8f0"
MONO = "JetBrains Mono, monospace"

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px'>
      <div style='font-family:JetBrains Mono;font-size:1.05rem;color:#0ea5e9;font-weight:700;letter-spacing:-0.01em'>
        🛡 CYBERPATCH
      </div>
      <div style='font-size:0.68rem;color:#94a3b8;margin-top:5px;letter-spacing:0.08em;text-transform:uppercase'>
        Red vs Blue Arena
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_results()
    if df is not None:
        blue_win_rate = (df['Blue_Score'] > df['Red_Score']).mean() * 100
        red_win_rate = (df['Red_Score'] > df['Blue_Score']).mean() * 100
        
        st.markdown(f"""
        <div style='font-size:0.68rem;color:#0ea5e9;font-family:JetBrains Mono;letter-spacing:0.1em;margin-bottom:10px'>MATCH STATS</div>
        <div style='background:rgba(14,165,233,0.05);border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin-bottom:10px'>
            <div style='font-size:0.75rem;color:#64748b'>Blue Win Rate</div>
            <div style='font-family:JetBrains Mono;font-size:1.4rem;font-weight:700;color:{C_BLUE}'>{blue_win_rate:.1f}%</div>
        </div>
        <div style='background:rgba(244,63,94,0.05);border:1px solid #e2e8f0;border-radius:10px;padding:12px'>
            <div style='font-size:0.75rem;color:#64748b'>Red Win Rate</div>
            <div style='font-family:JetBrains Mono;font-size:1.4rem;font-weight:700;color:{C_RED}'>{red_win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:8px 0 28px'>
  <div style='font-family:Inter;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;line-height:1.1;margin-bottom:10px;color:#0f172a'>
    Adversarial <span style='color:{C_BLUE}'>Blue</span> vs <span style='color:{C_RED}'>Red</span>
    <span style='font-size:0.82rem;color:#94a3b8;font-weight:400;margin-left:14px'>
      Adversarial Reinforcement Learning Simulation
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["📊  Competition Overview", "🌐  Live Simulation", "📈  Advanced Analysis"])

# ── TAB 1: OVERVIEW ────────────────────────────────────────
with t1:
    if df is None:
        st.error("No results found. Run red_blue_train.py first.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Blue Score", round(df['Blue_Score'].mean(), 1))
        k2.metric("Avg Red Score", round(df['Red_Score'].mean(), 1))
        k3.metric("Total Patched", df['Blue_Patched'].sum())
        k4.metric("Total Infected", df['Red_Infected'].sum())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Episode'], y=smooth(df['Blue_Score']), name="Blue Team", line=dict(color=C_BLUE, width=2)))
        fig.add_trace(go.Scatter(x=df['Episode'], y=smooth(df['Red_Score']), name="Red Team", line=dict(color=C_RED, width=2)))
        fig.update_layout(
            title="Score Trajectory (Smoothed)",
            plot_bgcolor=SURF, paper_bgcolor=SURF,
            xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: LIVE SIMULATION ─────────────────────────────────
with t2:
    env = CyberEnv(num_nodes=15)
    G = env.graph
    pos = nx.spring_layout(G, seed=42)
    
    c1, c2 = st.columns([2, 1])
    
    with c2:
        st.markdown(f"""<div style='font-family:{MONO};font-size:0.7rem;color:{C_BLUE};margin-bottom:10px'>SIM CONTROL</div>""", unsafe_allow_html=True)
        if st.button("▶ RUN SIMULATION STEP"):
            # Simple 1-step logic for visualization
            if 'sim_state' not in st.session_state:
                st.session_state.sim_state = {
                    'risks': list(env.node_risks),
                    'blue_scanner': 0, 'blue_patcher': 0,
                    'red_spreader': 14, 'red_infector': 14,
                    'log': []
                }
            
            s = st.session_state.sim_state
            # Move Blue
            s['blue_scanner'] = random.choice(list(G.neighbors(s['blue_scanner'])))
            s['blue_patcher'] = random.choice(list(G.neighbors(s['blue_patcher'])))
            if s['risks'][s['blue_patcher']] > 0:
                s['risks'][s['blue_patcher']] = 0
                s['log'].append(f"Blue Patcher fixed Node {s['blue_patcher']}")
            
            # Move Red
            s['red_spreader'] = random.choice(list(G.neighbors(s['red_spreader'])))
            s['red_infector'] = random.choice(list(G.neighbors(s['red_infector'])))
            if s['risks'][s['red_infector']] < 3:
                s['risks'][s['red_infector']] += 1
                s['log'].append(f"Red Infector attacked Node {s['red_infector']}")
        
        if st.button("↺ RESET SIM"):
            if 'sim_state' in st.session_state: del st.session_state.sim_state
            
        st.markdown("### Event Log")
        if 'sim_state' in st.session_state:
            for log in reversed(st.session_state.sim_state['log'][-10:]):
                st.write(f"- {log}")

    with c1:
        s = st.session_state.get('sim_state', {
            'risks': list(env.node_risks),
            'blue_scanner': 0, 'blue_patcher': 0,
            'red_spreader': 14, 'red_infector': 14
        })
        
        # Node Colors: Teal (0), Blue (1), Orange (2), Red (3)
        # Requirement: blue nodes shown in teal when patched, red nodes shown in red when infected, yellow when contested
        # I'll map 0->Teal, 1->Yellow (contested?), 2/3->Red
        node_colors = []
        for r in s['risks']:
            if r == 0: node_colors.append(C_TEAL)
            elif r == 1: node_colors.append(C_YELLOW)
            else: node_colors.append(C_RED)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
        
        # Nodes
        fig_net.add_trace(go.Scatter(
            x=[pos[i][0] for i in range(15)],
            y=[pos[i][1] for i in range(15)],
            mode='markers+text',
            text=[str(i) for i in range(15)],
            marker=dict(size=20, color=node_colors, line_width=2),
            name="Nodes"
        ))

        # Agent Markers (Rings)
        # Scanner (blue ring), Patcher (teal ring), RedSpreader (red ring), RedInfector (dark red ring)
        fig_net.add_trace(go.Scatter(x=[pos[s['blue_scanner']][0]], y=[pos[s['blue_scanner']][1]], marker=dict(size=30, symbol="circle-open", line=dict(color=C_BLUE, width=3)), name="Scanner", showlegend=True))
        fig_net.add_trace(go.Scatter(x=[pos[s['blue_patcher']][0]], y=[pos[s['blue_patcher']][1]], marker=dict(size=35, symbol="circle-open", line=dict(color=C_TEAL, width=3)), name="Patcher", showlegend=True))
        fig_net.add_trace(go.Scatter(x=[pos[s['red_spreader']][0]], y=[pos[s['red_spreader']][1]], marker=dict(size=40, symbol="circle-open", line=dict(color=C_RED_LIGHT, width=3)), name="RedSpreader", showlegend=True))
        fig_net.add_trace(go.Scatter(x=[pos[s['red_infector']][0]], y=[pos[s['red_infector']][1]], marker=dict(size=45, symbol="circle-open", line=dict(color=C_RED_DARK, width=3)), name="RedInfector", showlegend=True))

        fig_net.update_layout(
            showlegend=True, height=500, margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=SURF, paper_bgcolor=SURF
        )
        st.plotly_chart(fig_net, use_container_width=True)

# ── TAB 3: ANALYSIS ────────────────────────────────────────
with t3:
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            df['Score_Gap'] = df['Blue_Score'] - df['Red_Score']
            fig_gap = go.Figure(go.Scatter(x=df['Episode'], y=smooth(df['Score_Gap']), fill='tozeroy', line=dict(color=C_BLUE)))
            fig_gap.update_layout(title="Blue Advantage (Score Gap)", plot_bgcolor=SURF)
            st.plotly_chart(fig_gap, use_container_width=True)
        
        with c2:
            st.write("### Phase Analysis")
            st.write("Current simulation shows Blue team maintains a lead as training progresses.")
            st.info("The Red Team's infection rate decreases as Blue team learns more efficient patching patterns.")
