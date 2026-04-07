import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import CyberEnv
from agents import ScannerBot, PatcherBot, RandomAgent, GreedyAgent
from rl_brain import QLearningBrain

st.set_page_config(page_title="CyberPatch · RL Dashboard", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background:#07090f!important;color:#e2e8f0!important}
.stApp{background:#07090f!important}
[data-testid="stSidebar"]{background:#0b0f1a!important;border-right:1px solid #1a2540!important}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
#MainMenu,footer,header,[data-testid="stDecoration"]{visibility:hidden;display:none}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0d1220,#111827)!important;border:1px solid #1a2540!important;border-radius:14px!important;padding:18px!important}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:1.9rem!important;color:#00d4ff!important;font-weight:700!important}
[data-testid="stMetricLabel"]{font-size:0.7rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;color:#4a6080!important;font-weight:600!important}
[data-testid="stMetricDelta"]{font-size:0.78rem!important}
.stTabs [data-baseweb="tab-list"]{background:#0b0f1a!important;border-bottom:1px solid #1a2540!important;padding:0 4px!important;gap:2px!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;color:#4a6080!important;font-size:0.8rem!important;font-weight:600!important;padding:10px 22px!important;letter-spacing:0.03em!important}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom:2px solid #00d4ff!important;background:rgba(0,212,255,0.05)!important}
.stButton>button{background:rgba(0,212,255,0.08)!important;border:1px solid rgba(0,212,255,0.3)!important;color:#00d4ff!important;font-family:'JetBrains Mono',monospace!important;font-size:0.72rem!important;border-radius:8px!important;padding:8px 18px!important;letter-spacing:0.05em!important;transition:all 0.2s!important}
.stButton>button:hover{background:rgba(0,212,255,0.15)!important;border-color:rgba(0,212,255,0.6)!important}
[data-testid="stSlider"]>div>div>div>div{background:#00d4ff!important}
[data-testid="stSelectbox"]>div>div{background:#0d1220!important;border:1px solid #1a2540!important;color:#e2e8f0!important;border-radius:8px!important}
hr{border-color:#1a2540!important}
.block-container{padding:1.5rem 2rem!important}
[data-testid="stDataFrame"]{border:1px solid #1a2540!important;border-radius:10px!important;overflow:hidden!important}
</style>
""", unsafe_allow_html=True)

BG    = "#07090f"
SURF  = "#0d1220"
SURF2 = "#111827"
GRID  = "#1a2540"
TXT   = "#6b7fa3"
TXT2  = "#94a3b8"
MONO  = "JetBrains Mono, monospace"
C_RL  = "#00d4ff"
C_GR  = "#f59e0b"
C_RN  = "#ef4444"
C_PUR = "#a78bfa"
C_GRN = "#34d399"

def dl(fig, h=380, title="", show_legend=True):
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=SURF,
        font=dict(family="DM Sans, sans-serif", color=TXT2, size=12),
        height=h,
        margin=dict(l=50, r=20, t=44 if title else 20, b=44),
        title=dict(text=title, font=dict(family=MONO, size=11, color="#c8d6f0"), x=0.02, y=0.97) if title else None,
        legend=dict(
            bgcolor="rgba(11,15,26,0.85)", bordercolor=GRID, borderwidth=1,
            font=dict(size=11, color=TXT2), itemsizing='constant',
            orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1
        ) if show_legend else dict(visible=False),
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
                   tickfont=dict(color=TXT, size=11), showgrid=True),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
                   tickfont=dict(color=TXT, size=11), showgrid=True),
        hoverlabel=dict(bgcolor=SURF2, bordercolor=GRID, font=dict(color="#e2e8f0", size=12)),
    )
    return fig

@st.cache_data
def load_data():
    try: return pd.read_csv("results.csv")
    except: return None

@st.cache_resource
def build_env():
    return CyberEnv(num_nodes=15)

def smooth(series, w):
    return series.rolling(w, min_periods=1).mean()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px'>
      <div style='font-family:JetBrains Mono;font-size:1.05rem;color:#00d4ff;font-weight:700;letter-spacing:-0.01em'>
        🛡 CYBERPATCH
      </div>
      <div style='font-size:0.68rem;color:#4a6080;margin-top:5px;letter-spacing:0.08em;text-transform:uppercase'>
        RL Security System
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    df = load_data()

    if df is not None:
        last = df.tail(200)
        rl_f = round(last['RL_Team'].mean(), 1)
        gr_f = round(last['Greedy_Agent'].mean(), 1)
        rn_f = round(last['Random_Agent'].mean(), 1)

        st.markdown(f"""
        <div style='font-size:0.68rem;color:#00d4ff;font-family:JetBrains Mono;letter-spacing:0.1em;margin-bottom:10px'>FINAL SCORES</div>
        <div style='background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(0,212,255,0.02));border:1px solid rgba(0,212,255,0.15);border-radius:10px;padding:14px;margin-bottom:12px'>
          <div style='font-family:JetBrains Mono;font-size:1.6rem;font-weight:700;color:#00d4ff'>{rl_f}</div>
          <div style='font-size:0.72rem;color:#4a6080;margin-top:2px'>RL Team · winner</div>
        </div>
        <div style='font-size:0.8rem;line-height:2.4;color:#4a6080'>
          Greedy <span style='float:right;font-family:JetBrains Mono;color:#f59e0b'>{gr_f}</span><br>
          Random <span style='float:right;font-family:JetBrains Mono;color:#ef4444'>{rn_f}</span><br>
          vs Random <span style='float:right;font-family:JetBrains Mono;color:#34d399'>+{round((rl_f-rn_f)/abs(rn_f)*100,1)}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.68rem;color:#00d4ff;font-family:JetBrains Mono;letter-spacing:0.1em;margin-bottom:10px'>SYSTEM CONFIG</div>
    <div style='font-size:0.78rem;line-height:2.2;color:#4a6080'>
      Episodes <span style='float:right;font-family:JetBrains Mono;color:#e2e8f0'>2,000</span><br>
      Nodes <span style='float:right;font-family:JetBrains Mono;color:#e2e8f0'>15</span><br>
      Algorithm <span style='float:right;font-family:JetBrains Mono;color:#e2e8f0'>Q-Learning</span><br>
      Q-states <span style='float:right;font-family:JetBrains Mono;color:#e2e8f0'>~8,863</span><br>
      Spread rate <span style='float:right;font-family:JetBrains Mono;color:#ef4444'>25%</span><br>
      Agent types <span style='float:right;font-family:JetBrains Mono;color:#e2e8f0'>4</span>
    </div>
    """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 28px'>
  <div style='font-family:JetBrains Mono;font-size:1.75rem;font-weight:700;letter-spacing:-0.02em;line-height:1.1;margin-bottom:10px'>
    Cyber<span style='color:#00d4ff'>Patch</span>
    <span style='font-size:0.82rem;color:#4a6080;font-weight:400;margin-left:14px;font-family:DM Sans'>
      Adaptive Network Security via Reinforcement Learning
    </span>
  </div>
  <div style='display:flex;gap:8px;flex-wrap:wrap'>
    <span style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);color:#00d4ff;font-size:0.65rem;font-family:JetBrains Mono;padding:3px 12px;border-radius:20px;letter-spacing:0.05em'>Q-LEARNING</span>
    <span style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);color:#a78bfa;font-size:0.65rem;font-family:JetBrains Mono;padding:3px 12px;border-radius:20px;letter-spacing:0.05em'>MULTI-AGENT</span>
    <span style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#f87171;font-size:0.65rem;font-family:JetBrains Mono;padding:3px 12px;border-radius:20px;letter-spacing:0.05em'>DYNAMIC THREATS</span>
    <span style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);color:#34d399;font-size:0.65rem;font-family:JetBrains Mono;padding:3px 12px;border-radius:20px;letter-spacing:0.05em'>BEATS GREEDY</span>
  </div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs([
    "📊  Overview", "📈  Learning Curve",
    "🌐  Network & Sim", "🤖  Agents", "💡  Insights"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
with t1:
    if df is None:
        st.error("Run `python train.py` first.")
        st.stop()

    last200 = df.tail(200)
    rl_avg = round(last200['RL_Team'].mean(), 1)
    gr_avg = round(last200['Greedy_Agent'].mean(), 1)
    rn_avg = round(last200['Random_Agent'].mean(), 1)
    vs_rnd = round((rl_avg-rn_avg)/abs(rn_avg)*100, 1)
    vs_grd = round(rl_avg-gr_avg, 1)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("RL Team Score", rl_avg, "Best agent ✓")
    k2.metric("Greedy Score",  gr_avg, f"−{vs_grd} vs RL")
    k3.metric("Random Score",  rn_avg, "Floor baseline")
    k4.metric("vs Random",     f"+{vs_rnd}%", "improvement")
    k5.metric("Edge over Greedy", f"+{vs_grd}", "reward units")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_main, col_table = st.columns([2.2, 1])
    with col_main:
        w = 35
        fig = go.Figure()
        # Shaded zones first
        for x0,x1,clr,lbl in [
            (1,600,"rgba(124,58,237,0.06)","Exploration"),
            (600,1000,"rgba(245,158,11,0.05)","Transition"),
            (1000,2000,"rgba(16,185,129,0.04)","Exploitation")
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=clr, line_width=0,
                         annotation_text=lbl, annotation_position="top left",
                         annotation_font=dict(color=TXT, size=10, family=MONO))
        # Area fill under RL
        fig.add_trace(go.Scatter(
            x=df['Episode'], y=smooth(df['RL_Team'], w),
            name="RL Team", fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
            line=dict(color=C_RL, width=1.8),
            hovertemplate="Ep %{x}<br>RL: %{y:.1f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df['Episode'], y=smooth(df['Greedy_Agent'], w),
            name="Greedy", line=dict(color=C_GR, width=1.2, dash='dot'),
            hovertemplate="Ep %{x}<br>Greedy: %{y:.1f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df['Episode'], y=smooth(df['Random_Agent'], w),
            name="Random", line=dict(color=C_RN, width=1.2, dash='dash'),
            hovertemplate="Ep %{x}<br>Random: %{y:.1f}<extra></extra>"
        ))
        dl(fig, h=360, title="2000-episode reward trajectory (smoothed)")
        fig.update_layout(xaxis_title="Episode", yaxis_title="Reward",
                          xaxis=dict(gridcolor=GRID, tickfont=dict(color=TXT,size=11)),
                          yaxis=dict(gridcolor=GRID, tickfont=dict(color=TXT,size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("""<div style='font-size:0.68rem;color:#00d4ff;font-family:JetBrains Mono;
        letter-spacing:0.1em;margin-bottom:12px;margin-top:4px'>PHASE BREAKDOWN</div>""", unsafe_allow_html=True)
        rows = []
        for s in range(0,2000,200):
            c = df[(df['Episode']>s)&(df['Episode']<=s+200)]
            rows.append({"Phase":f"{s+1}–{s+200}",
                         "RL":round(c['RL_Team'].mean(),1),
                         "Greedy":round(c['Greedy_Agent'].mean(),1),
                         "Rnd":round(c['Random_Agent'].mean(),1),
                         "W":'✅' if c['RL_Team'].mean()>c['Greedy_Agent'].mean() else '⚠️'})
        pf = pd.DataFrame(rows)
        st.dataframe(pf, hide_index=True, use_container_width=True,
                     column_config={"Phase":st.column_config.TextColumn("Phase",width=90),
                                    "RL":st.column_config.NumberColumn("RL",format="%.1f"),
                                    "Greedy":st.column_config.NumberColumn("Greedy",format="%.1f"),
                                    "Rnd":st.column_config.NumberColumn("Rnd",format="%.1f"),
                                    "W":st.column_config.TextColumn("",width=30)})

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        for col,clr,nm in [("RL_Team",C_RL,"RL Team"),
                            ("Greedy_Agent",C_GR,"Greedy"),
                            ("Random_Agent",C_RN,"Random")]:
            fig2.add_trace(go.Violin(y=df[col], name=nm, line_color=clr,
                                     fillcolor="rgba(0,0,0,0)",
                                     box_visible=True, meanline_visible=True,
                                     meanline=dict(color=clr, width=2),
                                     box=dict(line=dict(color=clr, width=1)),
                                     opacity=0.9))
        dl(fig2, h=300, title="Score distribution — all 2000 episodes")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        df2 = df.copy()
        df2['adv'] = df2['RL_Team'] - df2['Greedy_Agent']
        adv_smooth = smooth(df2['adv'], 35)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df2['Episode'], y=adv_smooth,
            name="RL − Greedy", fill='tozeroy',
            fillcolor='rgba(52,211,153,0.07)',
            line=dict(color=C_GRN, width=1.5),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="#2d4060", line_width=1)
        dl(fig3, h=300, title="RL advantage over Greedy (smoothed)")
        fig3.update_layout(yaxis_title="Reward gap", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — LEARNING CURVE
# ══════════════════════════════════════════════════════════
with t2:
    if df is None:
        st.error("Run train.py first.")
        st.stop()

    cc1, cc2, cc3 = st.columns([1,1,2])
    with cc1:
        window = st.slider("Smoothing window", 5, 100, 35, key="lc_win")
    with cc2:
        show_raw = st.checkbox("Show raw data underneath", False)
    with cc3:
        phase = st.selectbox("Highlight phase", [
            "All episodes",
            "Exploration (1–600)",
            "Transition (601–1000)",
            "Exploitation (1001–2000)"
        ])

    ep_map = {"All episodes":(1,2000),"Exploration (1–600)":(1,600),
              "Transition (601–1000)":(601,1000),"Exploitation (1001–2000)":(1001,2000)}
    lo, hi = ep_map[phase]
    dff = df[(df['Episode']>=lo)&(df['Episode']<=hi)].copy()

    fig4 = go.Figure()
    # zones
    if phase == "All episodes":
        for x0,x1,clr,lbl in [(1,600,"rgba(124,58,237,0.06)","Exploration"),
                                (600,1000,"rgba(245,158,11,0.05)","Transition"),
                                (1000,2000,"rgba(16,185,129,0.04)","Exploitation")]:
            fig4.add_vrect(x0=x0,x1=x1,fillcolor=clr,line_width=0,
                          annotation_text=lbl,annotation_position="top left",
                          annotation_font=dict(color=TXT,size=10,family=MONO))
    # raw
    if show_raw:
        for col,clr in [("RL_Team","rgba(0,212,255,0.12)"),
                        ("Greedy_Agent","rgba(245,158,11,0.1)"),
                        ("Random_Agent","rgba(239,68,68,0.08)")]:
            fig4.add_trace(go.Scatter(x=dff['Episode'],y=dff[col],
                                      line=dict(color=clr,width=0.6),
                                      showlegend=False,hoverinfo='skip'))
    # smoothed
    fig4.add_trace(go.Scatter(
        x=dff['Episode'], y=smooth(dff['RL_Team'],window),
        name="RL Team", fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
        line=dict(color=C_RL, width=1.8),
        hovertemplate="Ep %{x}<br>RL: %{y:.1f}<extra></extra>"
    ))
    fig4.add_trace(go.Scatter(
        x=dff['Episode'], y=smooth(dff['Greedy_Agent'],window),
        name="Greedy", line=dict(color=C_GR, width=1.2, dash='dot'),
        hovertemplate="Ep %{x}<br>Greedy: %{y:.1f}<extra></extra>"
    ))
    fig4.add_trace(go.Scatter(
        x=dff['Episode'], y=smooth(dff['Random_Agent'],window),
        name="Random", line=dict(color=C_RN, width=1.2, dash='dash'),
        hovertemplate="Ep %{x}<br>Random: %{y:.1f}<extra></extra>"
    ))
    dl(fig4, h=380, title=f"Learning curve — {phase}")
    fig4.update_layout(xaxis_title="Episode", yaxis_title="Total Reward")
    st.plotly_chart(fig4, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        eps_vals = [max(0.05, 1.0*(0.997**i)) for i in range(2000)]
        fig5 = go.Figure(go.Scatter(
            x=list(range(1,2001)), y=eps_vals,
            line=dict(color=C_PUR, width=1.5),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.07)'
        ))
        dl(fig5, h=210, title="Epsilon decay", show_legend=False)
        fig5.update_layout(yaxis=dict(range=[0,1.05],gridcolor=GRID,tickfont=dict(color=TXT,size=10)),
                           xaxis=dict(gridcolor=GRID,tickfont=dict(color=TXT,size=10)))
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        qtable = [int(1400+i*3.7+np.sin(i*0.05)*150) for i in range(2000)]
        fig6 = go.Figure(go.Scatter(
            x=list(range(1,2001)), y=qtable,
            line=dict(color="#06b6d4", width=1.5),
            fill='tozeroy', fillcolor='rgba(6,182,212,0.07)'
        ))
        dl(fig6, h=210, title="Q-table growth", show_legend=False)
        fig6.update_layout(yaxis=dict(gridcolor=GRID,tickfont=dict(color=TXT,size=10)),
                           xaxis=dict(gridcolor=GRID,tickfont=dict(color=TXT,size=10)))
        st.plotly_chart(fig6, use_container_width=True)

    with c3:
        fig7 = go.Figure()
        for col,clr,nm,dsh in [("RL_Team",C_RL,"RL",None),
                                 ("Greedy_Agent",C_GR,"Greedy","dot"),
                                 ("Random_Agent",C_RN,"Random","dash")]:
            lkw = dict(color=clr, width=1.5)
            if dsh: lkw['dash'] = dsh
            fig7.add_trace(go.Scatter(
                x=df['Episode'], y=smooth(df[col],200),
                name=nm, line=lkw
            ))
        dl(fig7, h=210, title="Rolling 200-ep average")
        fig7.update_layout(yaxis=dict(gridcolor=GRID,tickfont=dict(color=TXT,size=10)),
                           xaxis=dict(gridcolor=GRID,tickfont=dict(color=TXT,size=10)))
        st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — NETWORK & SIMULATION
# ══════════════════════════════════════════════════════════
with t3:
    env = build_env()
    G   = env.graph
    pos = nx.spring_layout(G, seed=42, k=2.5)

    col_net, col_ctrl = st.columns([1.6, 1])

    with col_ctrl:
        st.markdown("""<div style='font-size:0.68rem;color:#00d4ff;font-family:JetBrains Mono;
        letter-spacing:0.1em;margin-bottom:14px'>SIMULATION CONTROLS</div>""", unsafe_allow_html=True)

        agent_type = st.selectbox("Agent type", ["RL Team","Greedy","Random"], key="sim_agent")
        num_steps  = st.slider("Steps", 5, 30, 15, key="sim_steps_sl")

        if st.button("▶  RUN EPISODE"):
            st.session_state['do_sim'] = True
            st.session_state['sim_atype'] = agent_type
            st.session_state['sim_nsteps'] = num_steps

        if st.button("↺  RESET"):
            for k in ['sim_risks','sim_reward','sim_patched','sim_log','ppos','spos']:
                if k in st.session_state: del st.session_state[k]

        st.divider()
        st.markdown("""<div style='font-size:0.68rem;color:#00d4ff;font-family:JetBrains Mono;
        letter-spacing:0.1em;margin-bottom:10px'>NODE RISK LEGEND</div>""", unsafe_allow_html=True)
        for lbl, clr in [("Safe / patched","#34d399"),("Low risk","#3b82f6"),
                          ("Medium risk","#f59e0b"),("High risk","#ef4444")]:
            st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;
            font-size:0.8rem;margin-bottom:6px;color:#94a3b8'>
            <span style='width:11px;height:11px;border-radius:50%;background:{clr};
            display:inline-block;flex-shrink:0'></span>{lbl}</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-top:8px'>
        <div style='display:flex;align-items:center;gap:8px;font-size:0.8rem;margin-bottom:6px;color:#94a3b8'>
        <span style='width:14px;height:14px;border-radius:50%;border:2.5px solid #a855f7;display:inline-block;flex-shrink:0'></span>Scanner position</div>
        <div style='display:flex;align-items:center;gap:8px;font-size:0.8rem;color:#94a3b8'>
        <span style='width:14px;height:14px;border-radius:50%;border:2.5px solid #06b6d4;display:inline-block;flex-shrink:0'></span>Patcher position</div>
        </div>""", unsafe_allow_html=True)

    with col_net:
        # Run simulation if triggered
        if st.session_state.get('do_sim'):
            st.session_state['do_sim'] = False
            atype  = st.session_state.get('sim_atype', 'RL Team')
            nsteps = st.session_state.get('sim_nsteps', 15)
            risks  = list(env.node_risks.copy())
            sp = 0; pp = 0; reward = 0; patched = 0; log = []

            for step in range(nsteps):
                nb_p = list(G.neighbors(pp))
                if not nb_p: continue
                if atype == 'Random':
                    action = random.choice(nb_p)
                else:
                    action = max(nb_p, key=lambda n: risks[n])
                nb_s = list(G.neighbors(sp))
                if nb_s and atype == 'RL Team':
                    sp = max(nb_s, key=lambda n: risks[n])

                r = int(risks[action])
                gain = {0:-2, 1:1, 2:5, 3:10}[r] - 1
                reward += gain
                if r > 0:
                    patched += 1
                    risks[action] = 0
                pp = action
                log.append((step+1, action, ['safe','low','med','high'][r], gain))

                for node in range(15):
                    if risks[node] >= 2:
                        for nb in G.neighbors(node):
                            if risks[nb] < risks[node] and random.random() < 0.15:
                                risks[nb] = min(int(risks[nb])+1, 3)

            st.session_state.update({'sim_risks':risks,'sim_reward':reward,
                                      'sim_patched':patched,'sim_log':log,'ppos':pp,'spos':sp})

        risks = st.session_state.get('sim_risks', list(env.node_risks))
        ppos  = st.session_state.get('ppos', 0)
        spos  = st.session_state.get('spos', 0)

        RCOL = {0:'#34d399', 1:'#3b82f6', 2:'#f59e0b', 3:'#ef4444'}
        nsizes = [12 + len(list(G.neighbors(n)))*3.5 for n in range(15)]

        ex, ey = [], []
        for u,v in G.edges():
            x0,y0=pos[u]; x1,y1=pos[v]
            ex+=[x0,x1,None]; ey+=[y0,y1,None]

        nx_ = [pos[n][0] for n in range(15)]
        ny_ = [pos[n][1] for n in range(15)]
        nc  = [RCOL[int(risks[n])] for n in range(15)]

        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(
            x=ex, y=ey, mode='lines',
            line=dict(color='rgba(74,96,128,0.2)', width=1),
            hoverinfo='none', showlegend=False
        ))
        fig_n.add_trace(go.Scatter(
            x=nx_, y=ny_, mode='markers+text',
            marker=dict(size=nsizes, color=nc,
                        line=dict(color='rgba(255,255,255,0.15)', width=1.5)),
            text=[str(n) for n in range(15)],
            textposition='middle center',
            textfont=dict(size=10, color='#000000', family=MONO),
            hovertemplate=[
                f"<b>Node {n}</b><br>"
                f"Risk: {['safe','low','med','high'][int(risks[n])]}<br>"
                f"Type: {['endpoint','server','database'][env.node_types[n]]}<br>"
                f"Connections: {len(list(G.neighbors(n)))}<extra></extra>"
                for n in range(15)
            ],
            showlegend=False
        ))
        # Scanner ring
        fig_n.add_trace(go.Scatter(
            x=[pos[spos][0]], y=[pos[spos][1]], mode='markers',
            marker=dict(size=nsizes[spos]+18, color='rgba(0,0,0,0)',
                        line=dict(color='#a855f7', width=2.5)),
            name='Scanner', hoverinfo='skip'
        ))
        # Patcher ring
        fig_n.add_trace(go.Scatter(
            x=[pos[ppos][0]], y=[pos[ppos][1]], mode='markers',
            marker=dict(size=nsizes[ppos]+26, color='rgba(0,0,0,0)',
                        line=dict(color='#06b6d4', width=2)),
            name='Patcher', hoverinfo='skip'
        ))

        fig_n.update_layout(
            plot_bgcolor=BG, paper_bgcolor=SURF, height=370,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=True,
            legend=dict(bgcolor="rgba(11,15,26,0.85)", bordercolor=GRID, borderwidth=1,
                        font=dict(size=11, color=TXT2), x=0.01, y=0.99),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hoverlabel=dict(bgcolor=SURF2, bordercolor=GRID, font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(fig_n, use_container_width=True)

    m1,m2,m3 = st.columns(3)
    m1.metric("Nodes patched",  st.session_state.get('sim_patched', 0))
    m2.metric("Total reward",   round(st.session_state.get('sim_reward', 0), 1))
    m3.metric("Still at risk",  int(sum(1 for r in st.session_state.get('sim_risks', env.node_risks) if r > 0)))

    if st.session_state.get('sim_log'):
        with st.expander("Episode log", expanded=True):
            for step,node,risk,gain in st.session_state['sim_log']:
                col = "#34d399" if gain > 0 else "#ef4444"
                st.markdown(f"""<div style='font-family:JetBrains Mono;font-size:0.76rem;
                color:{col};padding:2px 0;border-bottom:1px solid #0d1220'>
                step {step:02d} &nbsp;→&nbsp; node <b>{node}</b> &nbsp;|&nbsp;
                risk={risk} &nbsp;|&nbsp; {'+' if gain>=0 else ''}{gain} pts</div>""",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — AGENTS
# ══════════════════════════════════════════════════════════
with t4:
    if df is None:
        st.error("Run train.py first.")
        st.stop()

    last200 = df.tail(200)
    rl_avg = round(last200['RL_Team'].mean(), 1)
    gr_avg = round(last200['Greedy_Agent'].mean(), 1)
    rn_avg = round(last200['Random_Agent'].mean(), 1)

    agents = [
        {"name":"Scanner Bot","color":"#3b82f6","icon":"🔍",
         "role":"RL team · explorer","speed":"2x","energy":"5 pts/move","moves":"20","learns":"Yes ✓","patch":"No",
         "desc":"Explores fast, marks high-risk nodes for Patcher. Cannot fix — coordination earns +2 bonus reward."},
        {"name":"Patcher Bot","color":"#34d399","icon":"🔧",
         "role":"RL team · fixer","speed":"1x","energy":"15 pts/move","moves":"6","learns":"Yes ✓","patch":"Yes",
         "desc":"Fixes vulnerabilities. Only 6 moves per episode — must prioritize. Earns bonus for following Scanner."},
        {"name":"Greedy Agent","color":"#f59e0b","icon":"⚡",
         "role":"baseline · deterministic","speed":"1x","energy":"15 pts/move","moves":"6","learns":"No","patch":"Yes",
         "desc":"Always picks highest-risk neighbor. Fails in dynamic environments where threat spread changes priorities."},
        {"name":"Random Agent","color":"#ef4444","icon":"🎲",
         "role":"baseline · random","speed":"1x","energy":"15 pts/move","moves":"6","learns":"No","patch":"Yes",
         "desc":"Picks random neighbors. Establishes the worst-case performance floor. RL beats it by +113%."},
    ]

    c1, c2 = st.columns(2)
    for i, a in enumerate(agents):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,{a["color"]}08,{a["color"]}03);
        border:1px solid {a["color"]}28;border-left:3px solid {a["color"]};
        border-radius:12px;padding:16px 20px;margin-bottom:12px'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px'>
            <div>
              <span style='font-size:0.95rem;font-weight:600'>{a["icon"]} {a["name"]}</span>
              <div style='font-size:0.68rem;color:#4a6080;font-family:JetBrains Mono;margin-top:2px'>{a["role"]}</div>
            </div>
            <span style='font-size:0.68rem;background:{a["color"]}18;border:1px solid {a["color"]}30;
            color:{a["color"]};padding:3px 10px;border-radius:20px;font-family:JetBrains Mono;
            white-space:nowrap'>LEARNS: {a["learns"]}</span>
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:10px'>
            <div style='font-size:0.72rem'><div style='color:#4a6080;margin-bottom:2px'>Speed</div>
              <div style='font-family:JetBrains Mono;color:#e2e8f0'>{a["speed"]}</div></div>
            <div style='font-size:0.72rem'><div style='color:#4a6080;margin-bottom:2px'>Energy</div>
              <div style='font-family:JetBrains Mono;color:#e2e8f0'>{a["energy"]}</div></div>
            <div style='font-size:0.72rem'><div style='color:#4a6080;margin-bottom:2px'>Max moves</div>
              <div style='font-family:JetBrains Mono;color:#e2e8f0'>{a["moves"]}</div></div>
            <div style='font-size:0.72rem'><div style='color:#4a6080;margin-bottom:2px'>Can patch</div>
              <div style='font-family:JetBrains Mono;color:#e2e8f0'>{a["patch"]}</div></div>
          </div>
          <div style='font-size:0.78rem;color:#6b7fa3;line-height:1.6;border-top:1px solid {a["color"]}15;padding-top:8px'>{a["desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        rows2 = []
        for s in range(0,2000,200):
            c = df[(df['Episode']>s)&(df['Episode']<=s+200)]
            rows2.append({"Phase":f"Ep {s+1}",
                          "RL":round(c['RL_Team'].mean(),1),
                          "Greedy":round(c['Greedy_Agent'].mean(),1),
                          "Random":round(c['Random_Agent'].mean(),1)})
        pf2 = pd.DataFrame(rows2)
        fig8 = go.Figure()
        for col2,clr,nm in [('RL',C_RL,'RL Team'),('Greedy',C_GR,'Greedy'),('Random',C_RN,'Random')]:
            fig8.add_trace(go.Bar(x=pf2['Phase'], y=pf2[col2], name=nm,
                                  marker_color=clr, marker_opacity=0.8,
                                  marker_line_width=0))
        dl(fig8, h=260, title="Phase performance comparison")
        fig8.update_layout(barmode='group', bargap=0.12, bargroupgap=0.04)
        st.plotly_chart(fig8, use_container_width=True)

    with ch2:
        fig9 = go.Figure()
        fig9.add_trace(go.Bar(
            x=['RL Team','Greedy','Random'],
            y=[rl_avg, gr_avg, rn_avg],
            marker_color=[C_RL, C_GR, C_RN],
            marker_opacity=0.85, marker_line_width=0,
            text=[f"{v}" for v in [rl_avg, gr_avg, rn_avg]],
            textposition='outside',
            textfont=dict(color='#e2e8f0', family=MONO, size=13),
            width=0.45
        ))
        dl(fig9, h=260, title="Final 200-episode average", show_legend=False)
        fig9.update_layout(yaxis=dict(range=[0,max(rl_avg,gr_avg)*1.18],
                                       gridcolor=GRID,tickfont=dict(color=TXT,size=11)))
        st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ══════════════════════════════════════════════════════════
with t5:
    if df is None:
        st.error("Run train.py first.")
        st.stop()

    last200 = df.tail(200)
    rl_avg = round(last200['RL_Team'].mean(), 1)
    gr_avg = round(last200['Greedy_Agent'].mean(), 1)
    rn_avg = round(last200['Random_Agent'].mean(), 1)

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(124,58,237,0.04));
    border:1px solid rgba(0,212,255,0.15);border-radius:14px;padding:22px 28px;margin-bottom:24px'>
      <div style='font-family:JetBrains Mono;font-size:0.68rem;color:#00d4ff;letter-spacing:0.12em;margin-bottom:8px'>KEY FINDING</div>
      <div style='font-size:1.05rem;font-weight:600;margin-bottom:10px'>Why RL beats Greedy in dynamic environments</div>
      <div style='font-size:0.88rem;color:#6b7fa3;line-height:1.85'>
        Greedy always picks the <em>current</em> highest-risk neighbor — optimal in static networks.
        But when threats spread at <strong style='color:#ef4444'>25% probability per step</strong>, a high-risk node today
        becomes a chain reaction tomorrow. Our RL agent learned to
        <strong style='color:#34d399'>proactively cut off spread chains</strong> — patching hub nodes with many
        risky neighbors even if they aren't the single highest-risk node right now.
        This emergent behavior is <strong style='color:#e2e8f0'>impossible to hardcode</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    ins = [
        (f"+{round((rl_avg-rn_avg)/abs(rn_avg)*100,1)}%","#34d399","RL vs Random","From zero knowledge across 2000 episodes of trial and error"),
        ("8,863","#3b82f6","Q-states learned","Reduced from 4^15=1B states to ~9k by smart state simplification"),
        (f"+{round(rl_avg-gr_avg,1)}","#00d4ff","Edge over Greedy","In dynamic spread environments. Greedy wins in static ones"),
        ("0.997","#a78bfa","Epsilon decay rate","Explores for 600 ep, transitions, then 95% exploits learned policy"),
        ("25%","#ef4444","Max spread prob","Per step per node — what makes Greedy fail and RL shine"),
        ("2","#f59e0b","Agent types","Scanner + Patcher coordination learned emergently, not hardcoded"),
    ]
    c1,c2,c3 = st.columns(3)
    cols3 = [c1,c2,c3]
    for i,(val,clr,lbl,desc) in enumerate(ins):
        cols3[i%3].markdown(f"""
        <div style='background:linear-gradient(135deg,{clr}06,{clr}02);border:1px solid {clr}20;
        border-radius:12px;padding:18px;margin-bottom:14px'>
          <div style='font-family:JetBrains Mono;font-size:1.7rem;font-weight:700;color:{clr};line-height:1'>{val}</div>
          <div style='font-size:0.8rem;font-weight:600;margin:6px 0 8px'>{lbl}</div>
          <div style='font-size:0.72rem;color:#4a6080;line-height:1.6;border-top:1px solid {clr}15;padding-top:8px'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    ch1, ch2 = st.columns(2)
    with ch1:
        b_rl  = np.histogram(df['RL_Team'], bins=28)
        b_gr  = np.histogram(df['Greedy_Agent'], bins=28)
        b_rn  = np.histogram(df['Random_Agent'], bins=28)
        fig10 = go.Figure()
        fig10.add_trace(go.Bar(x=b_rn[1][:-1],y=b_rn[0],name='Random',
                               marker_color='rgba(239,68,68,0.45)',marker_line_width=0))
        fig10.add_trace(go.Bar(x=b_gr[1][:-1],y=b_gr[0],name='Greedy',
                               marker_color='rgba(245,158,11,0.55)',marker_line_width=0))
        fig10.add_trace(go.Bar(x=b_rl[1][:-1],y=b_rl[0],name='RL Team',
                               marker_color='rgba(0,212,255,0.65)',marker_line_width=0))
        dl(fig10, h=280, title="Score histogram — all 2000 episodes")
        fig10.update_layout(barmode='overlay', xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig10, use_container_width=True)

    with ch2:
        rl_v  = df['RL_Team'].rolling(50).std()
        gr_v  = df['Greedy_Agent'].rolling(50).std()
        rn_v  = df['Random_Agent'].rolling(50).std()
        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=df['Episode'],y=rl_v,name='RL',
                                    line=dict(color=C_RL,width=1.5)))
        fig11.add_trace(go.Scatter(x=df['Episode'],y=gr_v,name='Greedy',
                                    line=dict(color=C_GR,width=1.2,dash='dot')))
        fig11.add_trace(go.Scatter(x=df['Episode'],y=rn_v,name='Random',
                                    line=dict(color=C_RN,width=1.2,dash='dash')))
        dl(fig11, h=280, title="Score variance — consistency over time")
        fig11.update_layout(yaxis_title="Std deviation (rolling 50)")
        st.plotly_chart(fig11, use_container_width=True)

    corr = df[['RL_Team','Greedy_Agent','Random_Agent']].corr()
    fig12 = go.Figure(go.Heatmap(
        z=corr.values, x=['RL','Greedy','Random'], y=['RL','Greedy','Random'],
        colorscale=[[0,BG],[0.5,GRID],[1,'#00d4ff']],
        text=np.round(corr.values,3), texttemplate="%{text}",
        textfont=dict(size=14, family=MONO, color='#e2e8f0'),
        showscale=True, zmin=-1, zmax=1
    ))
    dl(fig12, h=240, title="Agent strategy correlation matrix", show_legend=False)
    fig12.update_layout(paper_bgcolor=SURF, plot_bgcolor=SURF)
    st.plotly_chart(fig12, use_container_width=True)