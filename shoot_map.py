"""
Advanced Shot Map Analysis Pipeline.

This script fetches event data from the 2022 FIFA World Cup match between
Japan and Spain via StatsBomb Open Data, processes shot events, computes advanced
KPIs (Expected Goals), and renders a publication-ready tactical infographic.

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

# Match ID for Japan vs Spain (FIFA World Cup 2022 Group Stage)
MATCH_ID = 3857255

# Stream matching event data directly into memory
events = sb.events(match_id=MATCH_ID)

# =========================================================================
# 2. DATA ENGINEERING & QUANTITATIVE METRICS
# =========================================================================

# Isolate all raw shot events and optimize execution with explicit copy
shots = events[events['type'] == 'Shot'].copy()

# Handle missing data gracefully by filling unassigned xG values with zero
shots['shot_statsbomb_xg'] = shots['shot_statsbomb_xg'].fillna(0)

# Unpack spatial tracking array coordinates into explicit features
shots[['x', 'y']] = pd.DataFrame(shots['location'].tolist(), index=shots.index)

# Segment data frames by team cohorts
japan_shots = shots[shots['team'] == 'Japan']
spain_shots = shots[shots['team'] == 'Spain']

# Calculate high-level analytical performance metrics
japan_total_xg = japan_shots['shot_statsbomb_xg'].sum()
spain_total_xg = spain_shots['shot_statsbomb_xg'].sum()

japan_count = len(japan_shots)
spain_count = len(spain_shots)

japan_avg_xg = japan_total_xg / japan_count if japan_count > 0 else 0
spain_avg_xg = spain_total_xg / spain_count if spain_count > 0 else 0

japan_goals = len(japan_shots[japan_shots['shot_outcome'] == 'Goal'])
spain_goals = len(spain_shots[spain_shots['shot_outcome'] == 'Goal'])

# =========================================================================
# 3. VISUALIZATION ENVIRONMENT & FIELD ARCHITECTURE
# =========================================================================

# Initialize half-pitch grid system mimicking standard professional dimensions
pitch = Pitch(pitch_type='statsbomb', half=True, pitch_color='#0e1714', line_color='#4b5653')
fig, axs = pitch.grid(ncols=2, axis=False, endnote_height=0.05, title_height=0.12, 
                      grid_height=0.73, space=0.05)
fig.set_facecolor('#0e1714')

def draw_advanced_shots(df_team, ax, face_color, edge_color):
    """Iterate through shot events and map geometric markers to outcomes."""
    for _, shot in df_team.iterrows():
        outcome = shot['shot_outcome']
        xg = shot['shot_statsbomb_xg']
        
        # Scale marker size dynamically according to xG threat
        size = xg * 1100 + 120
        
        # Vectorized mapping of shot outcomes to visual markers
        if outcome == 'Goal':
            marker = '*'      # Star indicator for high tactical conversion
            alpha = 1.0
            size *= 1.5       # Emphasize goals visually on the canvas
        elif outcome == 'Blocked':
            marker = '^'      # Triangle indicator for shot obstruction
            alpha = 0.5       # Semi-transparent to showcase defensive density
        else:
            marker = 'o'      # Circle indicator for saved / wide attempts
            alpha = 0.7
            
        pitch.scatter(shot['x'], shot['y'], ax=ax, s=size, 
                      marker=marker, facecolors=face_color, edgecolors=edge_color, 
                      linewidth=1.2, alpha=alpha)

# Execute spatial mapping pipelines for both team clusters
draw_advanced_shots(japan_shots, axs['pitch'][0], '#2b6cb0', '#63b3ed')
draw_advanced_shots(spain_shots, axs['pitch'][1], '#feb2b2', '#e10600')

# =========================================================================
# 4. TACTICAL METRIC BOXES & LABELS
# =========================================================================

# Generate uniform KPI summaries using bounding boxes aligned at top-left (X=62, Y=5)
bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#1a202c', alpha=0.6, edgecolor='none')

japan_kpis = f"Shots: {japan_count}\nGoals: {japan_goals}\nTotal xG: {japan_total_xg:.2f}\nAvg xG: {japan_avg_xg:.2f}"
axs['pitch'][0].text(62, 5, japan_kpis, color='white', size=10, ha='left', va='top', bbox=bbox_props)

spain_kpis = f"Shots: {spain_count}\nGoals: {spain_goals}\nTotal xG: {spain_total_xg:.2f}\nAvg xG: {spain_avg_xg:.2f}"
axs['pitch'][1].text(62, 5, spain_kpis, color='white', size=10, ha='left', va='top', bbox=bbox_props)

# =========================================================================
# 5. METADATA, HEADERS, AND THE LEGEND
# =========================================================================

# Headings
axs['title'].text(0.5, 0.75, "ADVANCED SHOT MAP ANALYSIS", color='white', 
                  va='center', ha='center', fontsize=22, fontweight='bold')

sub_text = f"FIFA World Cup 2022 | Japan {japan_goals}-{spain_goals} Spain | Tactical Efficiency & xG Breakdown"
axs['title'].text(0.5, 0.30, sub_text, color='#a0aec0', va='center', ha='center', fontsize=11)

# Pitch Headers
axs['pitch'][0].set_title("Japan (Low Block & Efficient Transition)", color='#63b3ed', fontsize=12, fontweight='bold', pad=4)
axs['pitch'][1].set_title("Spain (High Possession & Positional Attack)", color='#fc8181', fontsize=12, fontweight='bold', pad=4)

# Render accurate coordinate markers in the endnote panel for the infographic legend
axs['endnote'].scatter(0.03, 0.5, s=150, marker='*', facecolors='white', edgecolors='white', transform=axs['endnote'].transAxes)
axs['endnote'].text(0.05, 0.5, "Goal", color='#a0aec0', va='center', ha='left', fontsize=10, transform=axs['endnote'].transAxes)

axs['endnote'].scatter(0.12, 0.45, s=100, marker='o', facecolors='none', edgecolors='white', linewidth=1.2, transform=axs['endnote'].transAxes)
axs['endnote'].text(0.14, 0.5, "Saved / Off Target", color='#a0aec0', va='center', ha='left', fontsize=10, transform=axs['endnote'].transAxes)

axs['endnote'].scatter(0.30, 0.45, s=100, marker='^', facecolors='white', edgecolors='white', alpha=0.5, transform=axs['endnote'].transAxes)
axs['endnote'].text(0.32, 0.5, "Blocked by Defense", color='#a0aec0', va='center', ha='left', fontsize=10, transform=axs['endnote'].transAxes)

# Footnote Credits
axs['endnote'].text(0.98, 0.5, "Created by: Cristiano Santos | StatsBomb Data", 
                    color='#718096', va='center', ha='right', fontsize=10, fontweight='bold')

plt.show()