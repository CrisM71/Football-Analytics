"""
Advanced Passing Network and Expected Threat (xT) Modeling Pipeline.

This script implements Karun Singh's original Expected Threat Markov-chain grid 
to evaluate ball-progression efficiency during the 2022 FIFA World Cup game 
between Japan and Spain. It filters for the clean pre-substitution window 
and dynamically maps nodes to xT values and line weights to passing frequency.

Author: Cristiano Santos
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from statsbombpy import sb

# =========================================================================
# 1. DATA INGESTION & PIPELINE SETUP
# =========================================================================

MATCH_ID = 3857255
events = sb.events(match_id=MATCH_ID)

# Isolate successful pass events
passes = events[(events['type'] == 'Pass') & (events['pass_outcome'].isna())].copy()

# Map spatial coordinates to structural feature matrices
passes[['x', 'y']] = pd.DataFrame(passes['location'].tolist(), index=passes.index)
passes[['end_x', 'end_y']] = pd.DataFrame(passes['pass_end_location'].tolist(), index=passes.index)

# Isolate tactical period before structural dilution (prior to first substitution)
substitutions = events[events['type'] == 'Substitution']
first_sub_minute = substitutions['minute'].min()
passes_clean = passes[passes['minute'] < first_sub_minute].copy()

# =========================================================================
# 2. KARUN SINGH'S EXPECTED THREAT (xT) MATRICIAL IMPLEMENTATION
# =========================================================================

# Standard 12x8 spatial threat values modeled via historical transition frequencies
XT_MATRIX = np.array([
    [0.006383, 0.007796, 0.009405, 0.010587, 0.012147, 0.014845, 0.017895, 0.021150, 0.024608, 0.028443, 0.033621, 0.038580],
    [0.007501, 0.008785, 0.010266, 0.011409, 0.013531, 0.016390, 0.019559, 0.023257, 0.027412, 0.033101, 0.040177, 0.042735],
    [0.007878, 0.009299, 0.010996, 0.012474, 0.015291, 0.018861, 0.023304, 0.028213, 0.034329, 0.043694, 0.054366, 0.056006],
    [0.008354, 0.009669, 0.011116, 0.013237, 0.016578, 0.020846, 0.026402, 0.032688, 0.042466, 0.057398, 0.076326, 0.078235],
    [0.008354, 0.009669, 0.011116, 0.013237, 0.016578, 0.020846, 0.026402, 0.032688, 0.042466, 0.057398, 0.076326, 0.078235],
    [0.007878, 0.009299, 0.010996, 0.012474, 0.015291, 0.018861, 0.023304, 0.028213, 0.034329, 0.043694, 0.054366, 0.056006],
    [0.007501, 0.008785, 0.010266, 0.011409, 0.013531, 0.016390, 0.019559, 0.023257, 0.027412, 0.033101, 0.040177, 0.042735],
    [0.006383, 0.007796, 0.009405, 0.010587, 0.012147, 0.014845, 0.017895, 0.021150, 0.024608, 0.028443, 0.033621, 0.038580]
])

def get_xt_value(x, y):
    """Interpolate continuous coordinates into discrete matrix indexes."""
    grid_x = int(np.clip(x / 10, 0, 11))
    grid_y = int(np.clip(y / 10, 0, 7))
    return XT_MATRIX[grid_y, grid_x]

# Calculate progression threat value changes across matrices
passes_clean['xt_start'] = passes_clean.apply(lambda row: get_xt_value(row['x'], row['y']), axis=1)
passes_clean['xt_end'] = passes_clean.apply(lambda row: get_xt_value(row['end_x'], row['end_y']), axis=1)
passes_clean['xt_delta'] = (passes_clean['xt_end'] - passes_clean['xt_start']).clip(lower=0)

# =========================================================================
# 3. CRITICAL ERROR-PREVENTION: SPANISH NICKNAME STANDARDIZATION
# =========================================================================

# Ensure database keys exist inside the current DataFrame schema to avoid KeyErrors
if 'player_nickname' not in passes_clean.columns:
    passes_clean['player_nickname'] = np.nan
if 'pass_recipient_nickname' not in passes_clean.columns:
    passes_clean['pass_recipient_nickname'] = np.nan

# Execute fallback validation logic
passes_clean['player_clean'] = passes_clean['player_nickname'].fillna(passes_clean['player'])
passes_clean['recipient_clean'] = passes_clean['pass_recipient_nickname'].fillna(passes_clean['pass_recipient'])

def get_clean_label(full_name):
    """Normalize long form database strings to standard media display labels."""
    special_names = {
        "Rodrigo Hernández Cascante": "Rodri",
        "Pedro González López": "Pedri",
        "Pablo Martín Páez Gavira": "Gavi",
        "Sergio Busquets i Burgos": "Busquets",
        "Daniel Olmo Carvajal": "Dani Olmo",
        "Unai Simón Mendibil": "Unai Simón",
        "César Azpilicueta Tanco": "Azpilicueta"
    }
    return special_names.get(full_name, full_name.split()[-1] if len(full_name.split()) > 1 else full_name)

# =========================================================================
# 4. NETWORK MODELING AGGREGATIONS
# =========================================================================

def calculate_xt_network(df_passes, team_name):
    """Extract spatial averages, network combinations, and cumulative threat maps."""
    team_df = df_passes[df_passes['team'] == team_name].copy()
    
    # Extract structural coordinate centers of gravity
    average_locs = team_df.groupby('player_clean').agg({'x': ['mean', 'count'], 'y': ['mean']})
    average_locs.columns = ['x', 'count', 'y']
    
    # Map quantitative offensive value creation attributes to nodes
    total_xt_per_player = team_df.groupby('player_clean')['xt_delta'].sum().reset_index(name='total_xt')
    average_locs = average_locs.join(total_xt_per_player.set_index('player_clean'))
    
    # Calculate combination matrix volumes for lines
    team_df['recipient_clean'] = team_df['recipient_clean'].fillna('')
    pass_combos = team_df.groupby(['player_clean', 'recipient_clean']).agg(
        pass_count=('type', 'size'),
        total_pair_xt=('xt_delta', 'sum')
    ).reset_index()
    
    pass_combos = pass_combos[pass_combos['player_clean'].isin(average_locs.index) & 
                              pass_combos['recipient_clean'].isin(average_locs.index)]
    return average_locs, pass_combos

japan_locs, japan_combos = calculate_xt_network(passes_clean, 'Japan')
spain_locs, spain_combos = calculate_xt_network(passes_clean, 'Spain')

# =========================================================================
# 5. GEOMETRIC ARCHITECTURE & PLOTTING
# =========================================================================

pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1714', line_color='#4b5653')
fig, axs = pitch.grid(ncols=2, axis=False, endnote_height=0.05, title_height=0.1, grid_height=0.75, space=0.05)
fig.set_facecolor('#0e1714')

# Draw Japan Channel
for _, row in japan_combos.iterrows():
    if row['pass_count'] > 1:
        p1, p2 = row['player_clean'], row['recipient_clean']
        alpha_val = np.clip(row['total_pair_xt'] * 30, 0.2, 0.9)
        pitch.lines(japan_locs.loc[p1, 'x'], japan_locs.loc[p1, 'y'],
                    japan_locs.loc[p2, 'x'], japan_locs.loc[p2, 'y'],
                    lw=row['pass_count'] * 0.8, color='#63b3ed', alpha=alpha_val, ax=axs['pitch'][0])

pitch.scatter(japan_locs['x'], japan_locs['y'], s=japan_locs['total_xt'] * 8000 + 100,
              color='#002f6c', edgecolors='#63b3ed', linewidth=2, alpha=0.9, ax=axs['pitch'][0])

for player, row in japan_locs.iterrows():
    axs['pitch'][0].text(row['x'], row['y'] + 2.5, get_clean_label(player), color='white', fontsize=8, ha='center', fontweight='bold')

# Draw Spain Channel
for _, row in spain_combos.iterrows():
    if row['pass_count'] > 3:
        p1, p2 = row['player_clean'], row['recipient_clean']
        alpha_val = np.clip(row['total_pair_xt'] * 15, 0.2, 0.9)
        pitch.lines(spain_locs.loc[p1, 'x'], spain_locs.loc[p1, 'y'],
                    spain_locs.loc[p2, 'x'], spain_locs.loc[p2, 'y'],
                    lw=row['pass_count'] * 0.3, color='#fc8181', alpha=alpha_val, ax=axs['pitch'][1])

pitch.scatter(spain_locs['x'], spain_locs['y'], s=spain_locs['total_xt'] * 4000 + 100,
              color='#e10600', edgecolors='#fc8181', linewidth=2, alpha=0.9, ax=axs['pitch'][1])

for player, row in spain_locs.iterrows():
    axs['pitch'][1].text(row['x'], row['y'] + 2.5, get_clean_label(player), color='white', fontsize=8, ha='center', fontweight='bold')

# =========================================================================
# 6. LABELS, EXPERT DETAILS & GRAPHICS
# =========================================================================

axs['title'].text(0.5, 0.75, "PASS NETWORK & EXPECTED THREAT (xT)", color='white', va='center', ha='center', fontsize=22, fontweight='bold')

details = f"Starting Tactical Phase (Until Minute {first_sub_minute}) | Node Size = Cumulative Player xT | Line Opacity = Threat"
axs['title'].text(0.5, 0.25, details, color='#a0aec0', va='center', ha='center', fontsize=11)

axs['pitch'][0].set_title("Japan (Low Block & Efficient Transitions)", color='#63b3ed', fontsize=12, fontweight='bold', pad=2)
axs['pitch'][1].set_title("Spain (High Possession & Positional Attack)", color='#fc8181', fontsize=12, fontweight='bold', pad=2)

axs['endnote'].text(0.02, 0.5, "Expected Threat (xT) model adapted from Karun Singh | StatsBomb Open Data", color='#718096', va='center', ha='left', fontsize=9)
axs['endnote'].text(0.98, 0.5, "Created by: Cristiano Santos", color='#718096', va='center', ha='right', fontsize=10, fontweight='bold')

plt.show()