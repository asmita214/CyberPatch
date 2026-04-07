import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from environment import CyberEnv
from agents import create_agent
import os

def plot_learning_curves():
    if not os.path.exists('results.csv'):
        print("results.csv not found. Please run train.py first.")
        return
        
    df = pd.read_csv('results.csv')
    
    # Apply moving average for smoother curves
    window = 20
    df_smooth = df.rolling(window, min_periods=1).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df_smooth['RL_Team'], label='RL Team (Scanner + Patcher)', color='blue', linewidth=2)
    plt.plot(df['Episode'], df_smooth['Random_Agent'], label='Random Agent', color='red', alpha=0.5)
    plt.plot(df['Episode'], df_smooth['Greedy_Agent'], label='Greedy Agent', color='orange', alpha=0.5)
    
    plt.title('Reward over 500 Episodes (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curve.png')
    print("Saved learning curve to learning_curve.png")

def plot_comparison():
    if not os.path.exists('results.csv'):
        return
        
    df = pd.read_csv('results.csv')
    
    rl_mean = df['RL_Team'].mean()
    rand_mean = df['Random_Agent'].mean()
    greedy_mean = df['Greedy_Agent'].mean()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['RL Team', 'Greedy Agent', 'Random Agent'], [rl_mean, greedy_mean, rand_mean], 
            color=['blue', 'orange', 'red'])
    plt.title('Average Reward over 500 Episodes')
    plt.ylabel('Average Reward')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
        
    plt.savefig('comparison_bar.png')
    print("Saved comparison chart to comparison_bar.png")

def live_network_graph(steps=30):
    """
    Shows a live animation of the network graph with the Greedy Agent
    moving around, changing the risks.
    """
    print("\nStarting live network graph demonstration (Greedy Agent)...")
    env = CyberEnv()
    agent = create_agent('greedy', start_node=0)
    
    plt.ion() # Interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(env.graph, seed=42)
    
    color_map = {
        0: 'lightgreen', # Patched
        1: 'yellow',     # Low
        2: 'orange',     # Medium
        3: 'red'         # High
    }
    
    shape_map = {
        0: 'o', # Endpoint
        1: 's', # Server
        2: '^'  # Database
    }
    
    state = env.reset()
    agent.reset(start_node=0)
    done = False
    
    for step in range(steps):
        if done:
            break
            
        ax.clear()
        
        node_colors = [color_map[env.node_risks[i]] for i in range(env.num_nodes)]
        
        # Plot nodes by type to apply different markers
        for n_type, m_shape in shape_map.items():
            nodes_of_type = [n for n in range(env.num_nodes) if env.node_types[n] == n_type]
            if nodes_of_type:
                colors_of_type = [color_map[env.node_risks[n]] for n in nodes_of_type]
                nx.draw_networkx_nodes(env.graph, pos, nodelist=nodes_of_type, node_color=colors_of_type, 
                                     node_shape=m_shape, node_size=700, ax=ax)
                
        nx.draw_networkx_edges(env.graph, pos, width=2, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(env.graph, pos, font_size=12, font_family="sans-serif", ax=ax)
        
        # Highlight Agent position
        nx.draw_networkx_nodes(env.graph, pos, nodelist=[agent.current_node], 
                             node_color='none', edgecolors='blue', linewidths=3, node_size=900, ax=ax)
        
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        
        legend_elements = [
            mpatches.Patch(color='red', label='High Risk'),
            mpatches.Patch(color='lightgreen', label='Safe/Patched'),
            mlines.Line2D([], [], color='blue', marker='o', markerfacecolor='none', markersize=15, 
                          markeredgewidth=3, label='Agent Position', linestyle='None'),
            mlines.Line2D([], [], color='gray', marker='o', linestyle='None', label='Endpoint'),
            mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='Server'),
            mlines.Line2D([], [], color='gray', marker='^', linestyle='None', label='Database'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_title(f'CyberPatch Live Mode | Step: {step+1}')
        ax.axis('off')
        
        plt.pause(0.5)
        
        # Agent move step
        action = agent.select_action(env)
        next_state, reward, done, info = env.step(agent.current_node, action, agent_type='greedy')
        agent.move(action)
        
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    try:
        plot_learning_curves()
        plot_comparison()
        live_network_graph()
    except Exception as e:
        print(f"Error visualizing: {e}")
