import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

from mirror_representative import filter_representatives

# Paths and constants
folder = "../data/optimised_code_19.02.2025/"
venn_data_file = "venn_data.csv"

# Mirror-class filtering. Each moral system and its Good/Bad relabelling
# (mirror) carry identical cooperation and stability indices, so plotting both
# double-counts every non-self-mirror system. With this flag on, every plot
# shows only the majority-good representative of each mirror class -- roughly
# half the space (1,048,576 -> 524,800 systems) -- as specified in the
# manuscript's Definition of majority-good representatives. See
# mirror_representative.py. All consistent discriminators and the Leading Eight
# are representatives at the studied error rates, so overlays are unaffected;
# only the background cloud thins.
FILTER_TO_REPRESENTATIVES = True

# Get all matching files in the directory
def find_matching_files(directory):
    pattern = r"global_complete_chi_.*_epsilon_.*\.csv$"
    files = []
    
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if re.search(pattern, full_path):
            files.append(full_path)
    
    return files

# Main execution

files = find_matching_files(folder)
    

# Load venn_data once (for labels and conditions)
venn_df = pd.read_csv(venn_data_file, low_memory=False)
if 'label' not in venn_df.columns:
    raise ValueError(f"Column 'label' not found in {venn_data_file}")
venn_df['label'] = venn_df['label'].fillna('')
if 'is_consistent_discriminating' not in venn_df.columns:
    raise ValueError(f"Column 'is_consistent_discriminating' not found in {venn_data_file}")
print(venn_df.columns.tolist())


# Prepare columns for merging
merge_cols = ['p', 'd', 'label', 'set_R1', 'set_R2', 'set_R3', 'set_R4', 'is_consistent_discriminating']

# Color palette for R sets
set_colors = {
    1: '#1a9850',  # R1
    2: '#7570b3',  # R2
    3: '#e6ab02',  # R3
    4: '#FF4040',  # R4
    0: 'gray'      # Not in any set
}

# # Action rule ↔ p-value mappings
# action_rule_p_values = {'DDDC': 8, 'CDDC': 9, 'DCDC': 10, 'CCDC': 11}
# p_to_action_rule      = {v: k for k, v in action_rule_p_values.items()}
# action_rule_mapping   = {'CCDC': 4, 'DCDC': 3, 'CDDC': 1, 'DDDC': 1}
# variant_colors        = {
#     'CCDC': '#FF4040',
#     'DCDC': '#e6ab02',
#     'CDDC': '#4daf4a',
#     'DDDC': '#005a32'
# }

# Action rule ↔ p-value mappings
action_rule_p_values = {'DDDC': 8, 'CDDC': 9, 'DCDC': 10, 'CCDC': 11}
p_to_action_rule      = {v: k for k, v in action_rule_p_values.items()}
action_rule_mapping   = {'CCDC': 4, 'DCDC': 3, 'CDDC': 1, 'DDDC': 1}
variant_colors        = {
    'CCDC': '#F86A8C',
    'DCDC': '#D6CCEA',
    'CDDC': '#FF00EA',
    'DDDC': '#A387EA'
}

# Updated mapping for the new legend labels
new_label_mapping = {
    8: "PDISC",
    9: "ODISC",
    10: "DISC",
    11: "CDISC"
}

# Offsets for label placement
label_offsets = {
    'L1':           [-0.01, 0.0],
    'L2':           [0.01, 0.00],
    'L3':           [-0.01, -0.005],
    'L4':           [0.01, 0.00],
    'L5':           [0.01005, 0.00],
    'L6':           [0.01, 0.00],
    'L7':           [-0.01, 0.01],
    'L8':           [-0.01, 0.005]
}
default_offset = (0.0005, 0.0005)

# Jitter settings
JITTER_AMOUNT = 0.005  # Standard deviation for jitter

# --- Composite-figure text scale -------------------------------------------
# Panels destined for the same composite are sized so that figure-inches ==
# final display-inches at DISPLAY_DPI px/in. Under this convention the
# apparent (on-screen) size of text is
#     apparent_px = font_pt / 72 * (target_px_width / figsize_width_in)
# and, when figsize_width_in == target_px_width / DISPLAY_DPI, that collapses
# to  apparent_px = font_pt * DISPLAY_DPI / 72 . So identical point sizes in
# two panels render at identical apparent size regardless of each panel's
# pixel width. Render dpi (savefig) affects only sharpness, never this.
#
# The zoom panel (plot_single_zoom_jittered) is the reference: it already sits
# at ~72 px/in (440 px / 6.15 in), so its point sizes below are inherited
# verbatim by the density panel.
DISPLAY_DPI = 72             # px per figure-inch in the assembled composite
AXIS_LABEL_PT = 12 * 1.8     # 21.6 pt  (matches zoom reference)
TICK_LABEL_PT = 10 * 1.8     # 18.0 pt
LEGEND_PT = 10 * 1.8         # 18.0 pt
ANNOT_PT = 15.5              # leader-line / arrow label size

def add_jitter(data, x_col, y_col, amount=JITTER_AMOUNT):
    """Add small random jitter to x and y coordinates."""
    data = data.copy()
    np.random.seed(42)  # For reproducibility
    data[f'{x_col}_jittered'] = data[x_col] + np.random.normal(0, amount, len(data))
    data[f'{y_col}_jittered'] = data[y_col] + np.random.normal(0, amount, len(data))
    return data

def merge_info(df):
    """Merge dataframe with venn_data, overwriting any conflicting columns with venn_data's version."""
    # Check all required columns exist in venn_df
    required_cols = ['p', 'd', 'label', 'set_R1', 'set_R2', 'set_R3', 'set_R4', 'is_consistent_discriminating']
    missing_cols = [col for col in required_cols if col not in venn_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing in venn_data.csv: {', '.join(missing_cols)}")
    
    # Identify potentially conflicting columns (columns in both dataframes)
    conflict_cols = [col for col in required_cols if col in df.columns and col not in ['p', 'd']]
    if conflict_cols:
        print(f"Note: Found overlapping columns: {conflict_cols}. These will be overwritten with venn_data values.")
        # Drop conflicting columns from the input dataframe
        df = df.drop(columns=conflict_cols)
    
    # Prepare subset for merging
    venn_subset = venn_df[required_cols].copy()
    
    # Merge with base dataframe
    merged = pd.merge(df, venn_subset, on=['p', 'd'], how='left')
    
    # Check for unmatched rows
    unmatched_rows = merged[merged['label'].isna()]
    if not unmatched_rows.empty:
        # Print first few unmatched p,d pairs for debugging
        unmatched_sample = unmatched_rows[['p', 'd']].head(5).values.tolist()
        raise ValueError(f"Found {len(unmatched_rows)} rows in data file with no matching entry in venn_data.csv. " 
                         f"Sample unmatched (p,d) pairs: {unmatched_sample}")
    
    # Calculate set_category
    merged['set_category'] = 0
    merged.loc[ merged.set_R1 & ~merged.set_R2 & ~merged.set_R3 & ~merged.set_R4, 'set_category'] = 1
    merged.loc[ merged.set_R1 &  merged.set_R2 & ~merged.set_R3 & ~merged.set_R4, 'set_category'] = 2
    merged.loc[ merged.set_R1 &  merged.set_R2 &  merged.set_R3 & ~merged.set_R4, 'set_category'] = 3
    merged.loc[ merged.set_R1 &  merged.set_R2 &  merged.set_R3 &  merged.set_R4, 'set_category'] = 4

    # Keep only one representative per mirror class (see FILTER_TO_REPRESENTATIVES).
    if FILTER_TO_REPRESENTATIVES:
        before = len(merged)
        merged = filter_representatives(merged)
        print(f"  Mirror filter: {before} -> {len(merged)} representative systems "
              f"({len(merged) / before:.1%} retained)")

    return merged
def annotate_labels(df, x_col, y_col):
    """Annotate non-empty labels with offsets."""
    for _, row in df[df['label'] != ''].iterrows():
        txt = row['label']
        dx, dy = label_offsets.get(txt, default_offset)
        ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
        va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
        plt.text(
            row[x_col] + dx,
            row[y_col] + dy,
            txt,
            fontsize=15,
            ha=ha,
            va=va,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )

def plot_all(by_area=False):
    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'
    
    # Get the default sizes and calculate new ones
    axis_label_size = 12 * 1.15  # Standard size * 1.15
    legend_font_size = 10 * 1.1  # Standard size * 1.1

    for file in files:
        df = pd.read_csv(file)
        
        # Extract chi and epsilon values from filename
        match = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", file)
        if match:
            chi = match.group(1)
            eps = match.group(2)

        merged = merge_info(df)

        # 1) Labeled vs. unlabeled
        plt.figure(figsize=(10, 8))
        sns.set_style('whitegrid')
        unlabeled = merged[merged['label'] == '']
        labeled   = merged[merged['label'] != '']
        sns.scatterplot(data=unlabeled, x='cooperation_index', y=y_col,
                        marker='o', s=25, color='gray', alpha=0.005, legend=False)
        sns.scatterplot(data=labeled,   x='cooperation_index', y=y_col,
                        marker='s', s=25, color='orange', alpha=0.8, legend=False)
        annotate_labels(labeled, 'cooperation_index', y_col)
        
        # Use explicit values for font sizes
        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        # No title
        plt.tight_layout()
        plt.savefig(f"scatterplot_labeled_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()

        # 2) R-set based
        plt.figure(figsize=(10, 8))
        
        none_points = merged[merged.set_category == 0]
        if not none_points.empty:
            sns.scatterplot(data=none_points, x='cooperation_index', y=y_col,
                          marker='o', s=25, color=set_colors[0], alpha=0.005, 
                          label='Not in any R set')
        
        for cat in [1, 2, 3, 4]:
            sub = merged[merged.set_category == cat]
            if not sub.empty:
                sns.scatterplot(data=sub, x='cooperation_index', y=y_col,
                              marker='o', s=25, color=set_colors[cat], alpha=0.5, 
                              label=f'Category {cat}')
                
        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        
        handles, _ = plt.gca().get_legend_handles_labels()
        legend_labels = [
            'Not in any R set',
            'R1 only',
            'R1 & R2 only',
            'R1 & R2 & R3 only',
            'All sets (R1–R4)'
        ]
        plt.legend(handles, legend_labels, loc='upper left', fontsize=legend_font_size)
        
        plt.tight_layout()
        plt.savefig(f"scatterplot_sets_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()

        # 3) Combined
        plt.figure(figsize=(10, 8))
        
        none_points = merged[merged.set_category == 0]
        if not none_points.empty:
            sns.scatterplot(data=none_points, x='cooperation_index', y=y_col,
                          marker='o', s=25, color=set_colors[0], alpha=0.005)
        
        for cat in [1, 2, 3, 4]:
            sub = merged[merged.set_category == cat]
            if not sub.empty:
                sns.scatterplot(data=sub, x='cooperation_index', y=y_col,
                              marker='o', s=25, color=set_colors[cat], alpha=0.5)
                
        annotate_labels(merged, 'cooperation_index', y_col)
        
        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles, legend_labels, loc='upper left', fontsize=legend_font_size)
        
        plt.tight_layout()
        plt.savefig(f"scatterplot_combined_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()


def plot_zoom(by_area=False):
    label_y = 'stability_index_area' if by_area else 'stability_index_uniform'
    
    # Set explicit font sizes
    axis_label_size = 12 * 1.8  # Standard size * 1.15
    legend_font_size = 10 * 1.8  # Standard size * 1.1
    legend_title_size = 11 * 1.5    # Standard size * 1.1

    for file in files:
        df = pd.read_csv(file)
        
        # Extract chi and epsilon values from filename
        match = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", file)
        if match:
            chi = match.group(1)
            eps = match.group(2)
            print(f"chi={chi}, ε={eps}")

        # Merge with venn_data to get the is_consistent_discriminating column
        merged = merge_info(df)
        
        # Now filter using the merged dataframe
        zoom_data = merged[merged.is_consistent_discriminating == True]
        if zoom_data.empty:
            print(f"No consistent discriminators for chi={chi}, ε={eps}")
            continue

        plt.figure(figsize=(10, 8))
        sns.set_style('whitegrid')
        zoom_data = zoom_data.copy()
        zoom_data['action_rule'] = zoom_data['p'].map(p_to_action_rule)

        # Dictionary to store handles for manual legend creation
        legend_handles = []
        legend_labels = []

        # Plot each action rule with p-values
        for p_val, rule in p_to_action_rule.items():
            subset = zoom_data[zoom_data['p'] == p_val]
            if not subset.empty:
                scatter = sns.scatterplot(
                    data=subset,
                    x='cooperation_index', y=label_y,
                    color=variant_colors[rule], s=25, marker='o',
                    label=new_label_mapping[p_val]
                )
                # Get the handle for this scatter plot
                handles, labels = scatter.get_legend_handles_labels()
                legend_handles.append(handles[-1])
                legend_labels.append(new_label_mapping[p_val])

        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        # Create a custom legend with the new labels
        plt.legend(legend_handles, legend_labels,  
                  loc='center left', fontsize=legend_font_size)
        plt.tight_layout()
        plt.savefig(f"scatterplot_zoom_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()


def plot_zoom_labeled(by_area=False, generous=False):
    label_y = 'stability_index_area' if by_area else 'stability_index_uniform'
    
    # Set explicit font sizes
    axis_label_size = 12 * 1.15  # Standard size * 1.15
    legend_font_size = 10 * 1.1  # Standard size * 1.1
    legend_title_size = 11 * 1.1  # Standard size * 1.1
    
    # Add custom offsets specifically for the zoom labeled plot
    zoom_label_offsets = {
        'L2': [0.0002,  0.0002],
        'L6': [0.0002,  0.0002],
        'GS': [0.0002,  0.0002]  # Adding offset for the new 'GS' label
    }

    for file in files:
        df = pd.read_csv(file)
        
        # Extract chi and epsilon values from filename
        match = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", file)
        if match:
            chi = match.group(1)
            eps = match.group(2)

        # Merge with venn_data to get all necessary columns
        merged = merge_info(df)
        
        # Now filter using the merged dataframe
        zoom_data = merged[merged.is_consistent_discriminating == True]
        if zoom_data.empty:
            print(f"No consistent discriminators for chi={chi}, ε={eps}")
            continue

        # Add the "GS" label if generous flag is set
        if generous:
            # Create a copy to avoid SettingWithCopyWarning
            zoom_data = zoom_data.copy()
            # Set label to "GS" where p=11 and d=55277
            gs_mask = (zoom_data['p'] == 11) & (zoom_data['d'] == 55277)
            zoom_data.loc[gs_mask, 'label'] = 'GS'
            print("Generous Standing (GS) label applied to matching rows")

        plt.figure(figsize=(10, 8))
        sns.set_style('whitegrid')
        zoom_data = zoom_data.copy()
        zoom_data['action_rule'] = zoom_data['p'].map(p_to_action_rule)

        # Dictionary to store handles for manual legend creation
        legend_handles = []
        legend_labels = []

        # Plot each action rule with p-values
        for p_val, rule in p_to_action_rule.items():
            subset = zoom_data[zoom_data['p'] == p_val]
            if not subset.empty:
                scatter = sns.scatterplot(
                    data=subset,
                    x='cooperation_index', y=label_y,
                    color=variant_colors[rule], s=25, marker='o',
                    label=new_label_mapping[p_val]
                )
                # Get the handle for this scatter plot
                handles, labels = scatter.get_legend_handles_labels()
                legend_handles.append(handles[-1])
                legend_labels.append(new_label_mapping[p_val])

        # Label L2, L6, and GS (if generous)
        label_set = ['L2', 'L6'] if not generous else ['L2', 'L6', 'GS']
        labeled_points = zoom_data[zoom_data['label'].isin(label_set)]
        
        for _, row in labeled_points.iterrows():
            txt = row['label']
            # Use zoom-specific offsets instead of the global ones
            dx, dy = zoom_label_offsets.get(txt, default_offset)
            ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
            va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
            plt.text(
                row['cooperation_index'] + dx,
                row[label_y] + dy,
                txt, fontsize=10.5, ha=ha, va=va,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )

        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        # Create a custom legend with the new labels
        plt.legend(legend_handles, legend_labels,  
                  loc='center left', fontsize=legend_font_size)
        
        plt.tight_layout()
        plt.savefig(f"scatterplot_zoom_labeled_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()

def plot_pareto(by_area=False, generous=False):
    """
    Plot points with pareto_front=True in scarlet and label those with non-empty labels.
    If generous=True, add "GS" label to points where p=11 and d=55277.
    """
    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'
    
    axis_label_size = 12 * 1.15
    legend_font_size = 10 * 1.1

    for file in files:
        df = pd.read_csv(file)
        
        match = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", file)
        if match:
            chi = match.group(1)
            eps = match.group(2)

        merged = merge_info(df)
        
        if 'pareto_front' not in merged.columns:
            print(f"Warning: 'pareto_front' column not found in {file}. Skipping.")
            continue

        # Add the "GS" label if generous flag is set
        if generous:
            # Create a copy to avoid SettingWithCopyWarning
            merged = merged.copy()
            # Set label to "GS" where p=11 and d=55277
            gs_mask = (merged['p'] == 11) & (merged['d'] == 55277)
            merged.loc[gs_mask, 'label'] = 'GS'
            print("Generous Standing (GS) label applied to matching rows")

        # Split into pareto and non-pareto points
        non_pareto = merged[merged['pareto_front'] == False]
        pareto = merged[merged['pareto_front'] == True]
        
        # Identify pareto points with non-empty labels
        labeled_pareto = pareto[pareto['label'] != '']
        
        plt.figure(figsize=(10, 8))
        sns.set_style('whitegrid')
        
        sns.scatterplot(data=non_pareto, x='cooperation_index', y=y_col,
                        marker='o', s=25, color='gray', alpha=0.005, legend=True, label='Non-Pareto')
        
        sns.scatterplot(data=pareto, x='cooperation_index', y=y_col,
                        marker='o', s=25, color='#FF2400', alpha=0.8, legend=True, label='Pareto Front')
        
        # Add labels for pareto points with non-empty labels
        for _, row in labeled_pareto.iterrows():
            txt = row['label']
            dx, dy = label_offsets.get(txt, default_offset)
            ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
            va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
            plt.text(
                row['cooperation_index'] + dx,
                row[y_col] + dy,
                txt,
                fontsize=10.5,
                ha=ha,
                va=va,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )
        
        plt.xlabel('cooperation index', fontsize=axis_label_size)
        plt.ylabel('stability index', fontsize=axis_label_size)
        
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        
        plt.legend(loc='upper left', fontsize=legend_font_size)
        plt.tight_layout()
        
        generous_tag = "_generous" if generous else ""
        plt.savefig(f"scatterplot_pareto{generous_tag}_chi_{chi}_epsilon_{eps}_area_{by_area}.png", dpi=300)
        plt.close()

def plot_single_labeled_jittered(chi, eps, by_area=False, jitter_amount=JITTER_AMOUNT):
    """
    Generate a single labeled scatterplot with jittering for specified chi/epsilon.
    Labels the Leading Eight (L1-L8).
    """
    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'
    axis_label_size = 12 * 1.8

    file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
    print(f"Loading {file_path}")

    df = pd.read_csv(file_path)
    merged = merge_info(df)

    # Add jitter
    merged = add_jitter(merged, 'cooperation_index', y_col, amount=jitter_amount)
    x_jittered = 'cooperation_index_jittered'
    y_jittered = f'{y_col}_jittered'

    plt.figure(figsize=(10, 10))  # Square aspect ratio
    sns.set_style('whitegrid')

    # Split into unlabeled and labeled (L1-L8)
    unlabeled = merged[merged['label'] == '']
    labeled = merged[merged['label'].str.startswith('L')]

    # Plot unlabeled points with jitter
    sns.scatterplot(data=unlabeled, x=x_jittered, y=y_jittered,
                    marker='o', s=25, color='gray', alpha=0.005, legend=False)

    # Plot labeled points with jitter
    sns.scatterplot(data=labeled, x=x_jittered, y=y_jittered,
                    marker='s', s=40, color='orange', alpha=0.9, legend=False)

    # Annotate L1-L8 labels (use original coordinates for label placement)
    for _, row in labeled.iterrows():
        txt = row['label']
        dx, dy = label_offsets.get(txt, default_offset)
        ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
        va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
        plt.text(
            row[x_jittered] + dx,
            row[y_jittered] + dy,
            txt,
            fontsize=15.5,
            ha=ha,
            va=va,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )

    plt.tick_params(axis='both', labelsize=10 * 1.8)  # Tick labels scaled by 1.5

    plt.xlabel('cooperation index', fontsize=axis_label_size)
    plt.ylabel('stability index', fontsize=axis_label_size)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()

    output_png = f"scatter_plots/scatterplot_large_chi_{chi}_epsilon_{eps}_area_{by_area}_jittered.png"
    plt.savefig(output_png, dpi=450)  # 10in * 450dpi = 4500px wide
    print(f"Saved {output_png}")
    plt.close()

GAME_STABILITY_COLS = {
    'PD': 'stability_index_PD',
    'SG': 'stability_index_SG',
    'SH': 'stability_index_SH',
}


# Colorbar placement for the density panel, as an axes-fraction [x, y, w, h]
# passed to ax.inset_axes. With the axes spanning -0.02..1.02 on both sides,
# a stability value v sits at fraction (v + 0.02) / 1.04.
#
# The aggregate panel's cloud leaves the top-left corner empty, so the key
# lives there. The per-game clouds do not: the stag hunt piles systems along
# sigma^SH = 1 across the whole width, and the prisoner's dilemma reaches high
# enough on the left to be nicked. Each game therefore parks the key in its own
# empty band, always flush left.
CBAR_INSET = [0.04, 0.94, 0.22, 0.018]
CBAR_INSET_BY_GAME = {
    'PD': [0.04, 0.788, 0.22, 0.018],   # sigma^PD ~ 0.8
    'SH': [0.04, 0.423, 0.22, 0.018],   # sigma^SH ~ 0.42
    'SG': list(CBAR_INSET),             # top-left corner is clear
}


def _cbar_inset(game=None, override=None):
    """Axes-fraction rectangle for the density panel's inset colorbar."""
    if override is not None:
        return list(override)
    if game is None:
        return list(CBAR_INSET)
    return list(CBAR_INSET_BY_GAME.get(game.upper(), CBAR_INSET))


def _stability_axis(by_area=False, game=None):
    """(column, y-axis label, filename tag) for the requested stability index.

    With `game=None` the caller gets the aggregate index -- uniform or
    area-weighted -- exactly as before. With `game` one of PD/SG/SH the
    game-specific column is used and the label carries the matching
    superscript (sigma^PD and friends), which is what distinguishes the
    disaggregated composites from the aggregate Fig. 2 one. `by_area` has no
    meaning for a single game, so combining the two is an error rather than a
    silently ignored argument.
    """
    if game is None:
        col = 'stability_index_area' if by_area else 'stability_index_uniform'
        label = (r'stability index ($\sigma$, area-weighted)' if by_area
                 else r'stability index ($\sigma$)')
        return col, label, f"_area_{by_area}"

    game = game.upper()
    if game not in GAME_STABILITY_COLS:
        raise ValueError(
            f"game must be one of {list(GAME_STABILITY_COLS)}, got {game!r}")
    if by_area:
        raise ValueError("by_area and game are mutually exclusive: the "
                         "game-specific indices are not area-weighted")
    label = rf'stability index ($\sigma^{{\mathrm{{{game}}}}}$)'
    return GAME_STABILITY_COLS[game], label, f"_{game.lower()}"


def _zoom_box(x, y, margin=0.06):
    """Bounding box (x0, x1, y0, y1) of the given points, padded by `margin`
    of each axis span.

    This single definition keeps the density panel's dashed call-out box and
    the zoom panel's viewport in lock-step: feed both the morally consistent
    discriminators and they cannot disagree about where the zoom region is,
    no matter how the error values move the red cloud around. A zero span (all
    points identical on an axis) falls back to a small fixed pad so the box
    never collapses to a line.
    """
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    xpad = (x1 - x0) * margin or 0.01
    ypad = (y1 - y0) * margin or 0.01
    return x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad


def plot_single_labeled_density(chi, eps, by_area=False,
                                 jitter_amount=JITTER_AMOUNT,
                                 gridsize=90, with_inset=False,
                                 out_dir="/tmp",
                                 l8_face='#C6FF00', l8_edge='#1A1A1A',
                                 l8_edge_lw=0.6, l8_marker='D', l8_size=22,
                                 draw_zoom_box=True, zoom_box_margin=0.06,
                                 file_suffix='', game=None, cbar_inset=None):
    """
    Publication-grade variant of plot_single_labeled_jittered.

    Renders the ~1M strategy cloud as a hexbin density layer (log-normalized,
    muted single-hue colormap), the morally consistent discriminators as a
    crimson scatter overlay, and the Leading Eight as gold-diamond markers
    with thin leader-line labels. Saves a PNG to out_dir and returns its path.

    `game` (PD/SG/SH) swaps the aggregate stability index for that game's own,
    labelling the axis sigma^PD and so on; everything else about the panel is
    unchanged, so the disaggregated composites read as siblings of Fig. 2.
    `cbar_inset` overrides the inset colorbar's axes-fraction rectangle, whose
    per-game default keeps the key off the data (see CBAR_INSET_BY_GAME).
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    y_col, y_label, name_tag = _stability_axis(by_area, game)

    file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    merged = merge_info(df)

    # Density layer: the background cloud only -- exclude every point that is
    # redrawn as an overlay (consistent discriminators as crimson dots, the
    # Leading Eight as gold diamonds) so nothing is plotted twice.
    bg = merged[(merged['is_consistent_discriminating'] != True)
                & ~merged['label'].astype(str).str.startswith('L')]
    cd = merged[merged['is_consistent_discriminating'] == True].copy()
    leading = merged[merged['label'].astype(str).str.startswith('L')].copy()

    # Jitter only the foreground points; hexbin handles its own aggregation.
    if not cd.empty:
        cd = add_jitter(cd, 'cooperation_index', y_col, amount=jitter_amount)
    if not leading.empty:
        leading = add_jitter(leading, 'cooperation_index', y_col,
                             amount=jitter_amount)

    # Composite sizing: figure-inches == final display-inches at DISPLAY_DPI,
    # so the shared point-size constants render at the same apparent size as
    # the zoom reference panel. The design details below (markers, line
    # weights, tick/colorbar geometry) were tuned at the original 6.5 in
    # width; `g` rescales them so the panel's look is unchanged while its
    # text jumps to the shared vocabulary.
    target_px = 720
    fig_in = target_px / DISPLAY_DPI          # 10.0 in
    g = fig_in / 6.5                          # design-detail scale vs 6.5 in

    # Style: clean white, Helvetica, embedded fonts.
    sns.set_style('white')
    prev_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 0.6 * g,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), constrained_layout=True)

    ax.grid(True, which='major', color='#EEEEEE', lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Custom muted single-hue colormap: paper-white -> cool slate.
    fog_cmap = LinearSegmentedColormap.from_list(
        'fog_slate',
        ['#FFFFFF', '#D9DEE4', '#9FA9B4', '#5C6B7A', '#2C3E50'],
        N=256,
    )

    hb = ax.hexbin(
        bg['cooperation_index'].values,
        bg[y_col].values,
        gridsize=gridsize,
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=fog_cmap,
        norm=LogNorm(vmin=1, vmax=max(10, len(bg) // 200)),
        mincnt=1,
        linewidths=0,
        edgecolors='none',
        zorder=1,
    )

    # Consistent discriminators: crimson with a thin white halo.
    if not cd.empty:
        n_cd = len(cd)
        if n_cd > 3000:
            face_alpha = 0.5
            edge_lw = 0.0
        else:
            face_alpha = 0.85
            edge_lw = 0.25
        ax.scatter(
            cd['cooperation_index_jittered'],
            cd[f'{y_col}_jittered'],
            s=10 * g**2, marker='o',
            facecolor='#C8102E', edgecolor='white',
            linewidth=edge_lw * g, alpha=face_alpha, zorder=3,
        )

    # Leading Eight: gold diamonds.
    if not leading.empty:
        ax.scatter(
            leading['cooperation_index_jittered'],
            leading[f'{y_col}_jittered'],
            s=l8_size * g**2, marker=l8_marker,
            facecolor=l8_face, edgecolor=l8_edge,
            linewidth=l8_edge_lw * g, zorder=4,
        )

        # Local override of label offsets — the new visual weight needs
        # different placement than the original label_offsets dict.
        density_label_offsets = {
            'L1': (-0.045,  0.020),
            'L2': ( 0.040,  0.025),
            'L3': (-0.020, -0.050),
            'L4': ( 0.040,  0.040),
            'L5': ( 0.055,  0.010),
            'L6': ( 0.040, -0.025),
            'L7': (-0.020,  0.050),
            'L8': (-0.040, -0.035),
        }
        for _, row in leading.iterrows():
            txt = row['label']
            if not isinstance(txt, str) or not txt.startswith('L'):
                continue
            dx, dy = density_label_offsets.get(txt, (0.03, 0.03))
            xa = row['cooperation_index_jittered']
            ya = row[f'{y_col}_jittered']
            ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
            va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
            ax.annotate(
                txt,
                xy=(xa, ya),
                xytext=(xa + dx, ya + dy),
                fontsize=ANNOT_PT, ha=ha, va=va,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5 * g),
                arrowprops=dict(
                    arrowstyle='-',
                    color='#1A1A1A',
                    lw=0.55 * g,
                    shrinkA=0, shrinkB=2 * g,
                ),
                zorder=5,
            )

    # Zoom call-out: a grey dashed rectangle around the consistent
    # discriminators -- exactly the region the right-hand zoom panel displays.
    # Derived from the same points as the zoom viewport (see _zoom_box), so it
    # follows the red cloud automatically as chi/eps shift it about.
    if draw_zoom_box and not cd.empty:
        from matplotlib.patches import Rectangle
        bx0, bx1, by0, by1 = _zoom_box(
            cd['cooperation_index'], cd[y_col], zoom_box_margin)
        ax.add_patch(Rectangle(
            (bx0, by0), bx1 - bx0, by1 - by0,
            fill=False, edgecolor='#808080', linestyle=(0, (6, 4)),
            linewidth=1.1 * g, zorder=6,
        ))

    sns.despine(ax=ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'cooperation index ($\kappa$)', fontsize=AXIS_LABEL_PT)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_PT)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_PT,
                   length=3.5 * g, width=0.5 * g)

    # Inset colorbar (horizontal, flush left). Decorative — kept at its
    # original apparent size by scaling its geometry and text with g. Its
    # height on the axes varies by game so it never sits over the cloud;
    # see CBAR_INSET_BY_GAME.
    cax = ax.inset_axes(_cbar_inset(game, cbar_inset))
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.4 * g)
    cb.ax.tick_params(labelsize=7 * g * 1.1, length=2 * g, width=0.4 * g, pad=1.5 * g)
    cb.set_label('systems / hex (log)', fontsize=7.5 * g * 1.1, labelpad=2 * g)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(
        out_dir,
        f"scatterplot_density_chi_{chi}_epsilon_{eps}{name_tag}"
        f"{file_suffix}.png",
    )
    fig.savefig(out_png, dpi=450)
    print(f"Saved {out_png}")
    plt.close(fig)
    mpl.rcParams.update(prev_rc)
    return out_png


def plot_panel_4x4_density(by_area=False, jitter_amount=0,
                             gridsize=75, out_dir='density_scatter'):
    """
    4x4 facet of density plots: rows = chi, cols = epsilon.
    Same rendering approach as plot_single_labeled_density, compressed
    for small-multiple panels (no per-cell L-labels, one shared colorbar).

    Jitter defaults to 0: the publication version of these panels plots
    systems at their exact coordinates, so the discrete stability/cooperation
    lattice reads cleanly. Pass jitter_amount>0 only for exploratory views.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    chi_values = [0.01, 0.02, 0.05, 0.1]
    epsilon_values = [0.0, 0.01, 0.02, 0.1]

    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'

    sns.set_style('white')
    prev_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fog_cmap = LinearSegmentedColormap.from_list(
        'fog_slate',
        ['#FFFFFF', '#D9DEE4', '#9FA9B4', '#5C6B7A', '#2C3E50'],
        N=256,
    )

    fig, axes = plt.subplots(
        4, 4, figsize=(11, 11),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    last_hb = None

    for row, chi in enumerate(chi_values):
        for col, eps in enumerate(epsilon_values):
            ax = axes[row, col]
            file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#888')
                continue

            merged = merge_info(df)
            # Background cloud only: drop overlay points (consistent
            # discriminators + Leading Eight) so none are plotted twice.
            bg = merged[(merged['is_consistent_discriminating'] != True)
                        & ~merged['label'].astype(str).str.startswith('L')]
            cd = merged[merged['is_consistent_discriminating'] == True].copy()
            leading = merged[merged['label'].astype(str).str.startswith('L')].copy()
            if not cd.empty:
                cd = add_jitter(cd, 'cooperation_index', y_col,
                                amount=jitter_amount)
            if not leading.empty:
                leading = add_jitter(leading, 'cooperation_index', y_col,
                                     amount=jitter_amount)

            ax.grid(True, which='major', color='#EEEEEE', lw=0.3, zorder=0)
            ax.set_axisbelow(True)

            hb = ax.hexbin(
                bg['cooperation_index'].values,
                bg[y_col].values,
                gridsize=gridsize, extent=(0.0, 1.0, 0.0, 1.0),
                cmap=fog_cmap,
                norm=LogNorm(vmin=1, vmax=max(10, len(bg) // 200)),
                mincnt=1, linewidths=0, edgecolors='none', zorder=1,
            )
            last_hb = hb

            if not cd.empty:
                n_cd = len(cd)
                if n_cd > 3000:
                    face_alpha, edge_lw = 0.5, 0.0
                else:
                    face_alpha, edge_lw = 0.85, 0.2
                ax.scatter(
                    cd['cooperation_index_jittered'],
                    cd[f'{y_col}_jittered'],
                    s=6, marker='o',
                    facecolor='#C8102E', edgecolor='white',
                    linewidth=edge_lw, alpha=face_alpha, zorder=3,
                )

            if not leading.empty:
                ax.scatter(
                    leading['cooperation_index_jittered'],
                    leading[f'{y_col}_jittered'],
                    s=14, marker='D',
                    facecolor='#C6FF00', edgecolor='#1A1A1A',
                    linewidth=0.5, zorder=4,
                )

            sns.despine(ax=ax)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', labelsize=8, length=2.5, width=0.4)

            if row == 0:
                ax.set_title(rf"$\varepsilon = {eps}$", fontsize=11)
            if col == 0:
                ax.set_ylabel(rf"$\chi = {chi}$", fontsize=11)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')

    # Shared colorbar along the top of the grid.
    if last_hb is not None:
        cb = fig.colorbar(
            last_hb, ax=axes, location='top',
            shrink=0.35, aspect=35, pad=0.02,
        )
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=8, length=2.5, width=0.4, pad=1.5)
        cb.set_label('systems / hex (log)', fontsize=9, labelpad=3)

    # Figure-wide axis labels.
    y_label = 'stability index (area-weighted)' if by_area else 'stability index'
    fig.supxlabel('cooperation index', fontsize=12)
    fig.supylabel(y_label, fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"density_panel_4x4_area_{by_area}.png")
    fig.savefig(out_png, dpi=450)
    print(f"Saved {out_png}")
    plt.close(fig)
    mpl.rcParams.update(prev_rc)


def plot_single_zoom_jittered(chi, eps, by_area=False, jitter_amount=JITTER_AMOUNT/5,
                              zoom_box_margin=0.06,
                              out_dir="scatter_plots", file_suffix="_jittered",
                              game=None):
    """
    Generate a single zoom scatterplot with jittering for specified chi/epsilon.
    Shows only consistent discriminating strategies, colored by action rule.
    Legend markers are 50% larger than plot markers.

    `jitter_amount=0` disables jitter entirely; `out_dir`/`file_suffix` let a
    caller park such a variant somewhere other than the canonical output.
    `game` (PD/SG/SH) plots that game's stability index instead of the
    aggregate one, matching the density panel it is stitched beside.

    Returns the output path, or None when the parameter pair has no consistent
    discriminators to zoom into.
    """
    y_col, y_label, name_tag = _stability_axis(by_area, game)
    axis_label_size = 12 * 1.8
    legend_font_size = 10 * 1.8

    marker_size = 25  # Plot marker size
    legend_marker_size = marker_size * 3  # 100% larger for legend

    file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
    print(f"Loading {file_path}")

    df = pd.read_csv(file_path)
    merged = merge_info(df)

    # Filter to consistent discriminating strategies
    zoom_data = merged[merged.is_consistent_discriminating == True].copy()
    if zoom_data.empty:
        print(f"No consistent discriminators for chi={chi}, ε={eps}")
        return None

    # Add jitter
    zoom_data = add_jitter(zoom_data, 'cooperation_index', y_col, amount=jitter_amount)
    x_jittered = 'cooperation_index_jittered'
    y_jittered = f'{y_col}_jittered'

    zoom_data['action_rule'] = zoom_data['p'].map(p_to_action_rule)

    plt.figure(figsize=(10*8/13, 10))  # 8:13 aspect ratio, same height as large plot
    sns.set_style('whitegrid')

    # Identify L2 and L6 points for arrow labels
    label_set = ['L2', 'L6']
    labeled_points = zoom_data[zoom_data['label'].isin(label_set)]

    # Plot each action rule with p-values (including L2 and L6)
    for p_val, rule in p_to_action_rule.items():
        subset = zoom_data[zoom_data['p'] == p_val]
        if not subset.empty:
            plt.scatter(
                subset[x_jittered], subset[y_jittered],
                c=variant_colors[rule], s=marker_size, marker='o',
                label=new_label_mapping[p_val], alpha=0.8
            )

    # Add L2 and L6 arrow labels
    zoom_label_offsets = {
        'L2': [-0.003, 0.003],
        'L6': [-0.003, 0.003],
    }

    for _, row in labeled_points.iterrows():
        txt = row['label']
        dx, dy = zoom_label_offsets.get(txt, default_offset)
        ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
        va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
        plt.annotate(
            txt,
            xy=(row[x_jittered], row[y_jittered]),
            xytext=(row[x_jittered] + dx, row[y_jittered] + dy),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            fontsize=15.5,
            ha=ha,
            va=va,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
        )

    plt.xlabel(r'cooperation index ($\kappa$)', fontsize=axis_label_size)
    plt.ylabel(y_label, fontsize=axis_label_size)
    plt.tick_params(axis='both', labelsize=10 * 1.8)  # Tick labels scaled by 1.5

    # Create legend with larger markers
    legend = plt.legend(loc='center left', fontsize=legend_font_size)
    for handle in legend.legend_handles:
        handle.set_sizes([legend_marker_size])

    # Pin the viewport to the same box the density panel draws its dashed
    # call-out from, so panel b shows exactly what panel a's rectangle frames.
    bx0, bx1, by0, by1 = _zoom_box(
        zoom_data['cooperation_index'], zoom_data[y_col], zoom_box_margin)
    plt.xlim(bx0, bx1)
    plt.ylim(by0, by1)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    output_png = os.path.join(
        out_dir,
        f"scatterplot_zoom_chi_{chi}_epsilon_{eps}{name_tag}{file_suffix}.png",
    )
    plt.savefig(output_png, dpi=450)
    print(f"Saved {output_png}")
    plt.close()
    return output_png

def plot_panel_4x4_zoom(by_area=False, jitter_amount=JITTER_AMOUNT/5,
                        zoom_lim=(0.6, 1.0), out_dir='scatter_plots'):
    """
    4x4 facet of the zoom scatterplot: rows = chi, cols = epsilon.

    Each cell reproduces plot_single_zoom_jittered's content -- only the
    morally consistent discriminating strategies, coloured by action rule --
    but the cells are square and every one plots the same fixed region of
    parameter space (zoom_lim x zoom_lim, kappa on x, sigma on y) so the panels
    are directly comparable. L2 and L6 (the Leading Eight members that are
    consistent discriminators) are arrow-labelled in each cell. A single shared
    legend serves the whole figure.
    """
    from matplotlib.ticker import MultipleLocator

    chi_values = [0.01, 0.02, 0.05, 0.1]
    epsilon_values = [0.0, 0.01, 0.02, 0.1]

    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'

    marker_size = 10          # per-cell plot marker
    legend_marker_size = 60   # shared-legend marker
    lo, hi = zoom_lim
    span = hi - lo

    # Arrow-label offsets, scaled to the zoom span so they read the same in
    # each cell regardless of the fixed window.
    zoom_label_offsets = {
        'L2': [-0.08 * span,  0.08 * span],
        'L6': [-0.08 * span,  0.08 * span],
    }

    sns.set_style('whitegrid')

    fig, axes = plt.subplots(
        4, 4, figsize=(12, 12),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    legend_handles, legend_labels = [], []

    for row, chi in enumerate(chi_values):
        for col, eps in enumerate(epsilon_values):
            ax = axes[row, col]
            file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#888')
                ax.set_box_aspect(1)
                continue

            merged = merge_info(df)
            zoom_data = merged[merged.is_consistent_discriminating == True].copy()
            if not zoom_data.empty:
                zoom_data = add_jitter(zoom_data, 'cooperation_index', y_col,
                                       amount=jitter_amount)
            x_jittered = 'cooperation_index_jittered'
            y_jittered = f'{y_col}_jittered'

            # Scatter each action rule.
            for p_val, rule in p_to_action_rule.items():
                subset = zoom_data[zoom_data['p'] == p_val] if not zoom_data.empty else zoom_data
                if not subset.empty:
                    h = ax.scatter(
                        subset[x_jittered], subset[y_jittered],
                        c=variant_colors[rule], s=marker_size, marker='o',
                        alpha=0.8, label=new_label_mapping[p_val],
                    )
                    if new_label_mapping[p_val] not in legend_labels:
                        legend_handles.append(h)
                        legend_labels.append(new_label_mapping[p_val])

            # Arrow-label L2 and L6.
            if not zoom_data.empty:
                labeled_points = zoom_data[zoom_data['label'].isin(['L2', 'L6'])]
                for _, pt in labeled_points.iterrows():
                    txt = pt['label']
                    dx, dy = zoom_label_offsets.get(txt, default_offset)
                    ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
                    va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
                    ax.annotate(
                        txt,
                        xy=(pt[x_jittered], pt[y_jittered]),
                        xytext=(pt[x_jittered] + dx, pt[y_jittered] + dy),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                        fontsize=10, ha=ha, va=va,
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=1.5),
                    )

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_box_aspect(1)
            # Vertical-axis gridlines only every 0.1 (drop the 0.05 lines).
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.tick_params(axis='both', labelsize=9)

            if row == 0:
                ax.set_title(rf"$\varepsilon = {eps}$", fontsize=12)
            if col == 0:
                ax.set_ylabel(rf"$\chi = {chi}$", fontsize=12)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')

    # Single shared legend, outside the grid to the right.
    if legend_handles:
        leg = fig.legend(
            legend_handles, legend_labels,
            loc='outside center right', fontsize=13,
            markerscale=1, frameon=True,
        )
        for handle in leg.legend_handles:
            handle.set_sizes([legend_marker_size])

    # Figure-wide axis labels.
    fig.supxlabel(r'cooperation index ($\kappa$)', fontsize=14)
    fig.supylabel(
        r'stability index ($\sigma$, area-weighted)' if by_area
        else r'stability index ($\sigma$)',
        fontsize=14,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"scatterplot_panel_4x4_zoom_area_{by_area}.png")
    fig.savefig(out_png, dpi=450)
    print(f"Saved {out_png}")
    plt.close(fig)

def plot_sweep_labeled(by_area=False, jitter_amount=JITTER_AMOUNT):
    """
    Generate labeled scatterplots for a sweep of chi and epsilon values.
    Epsilon in {0.0, 0.01, 0.02, 0.1}, Chi in {0.01, 0.02, 0.05, 0.1}.
    """
    epsilon_values = [0.0, 0.01, 0.02, 0.1]
    chi_values = [0.01, 0.02, 0.05, 0.1]

    for chi in chi_values:
        for eps in epsilon_values:
            try:
                plot_single_labeled_jittered(chi=chi, eps=eps, by_area=by_area, jitter_amount=jitter_amount)
            except FileNotFoundError as e:
                print(f"Skipping chi={chi}, eps={eps}: {e}")

def plot_sweep_labeled_small(by_area=False, jitter_amount=JITTER_AMOUNT, datahighlight="leadingeight"):
    """
    Generate small labeled scatterplots (3.25x3.25 in) for a sweep of chi and epsilon values.
    Suitable for tiling into a montage.
    Epsilon in {0.0, 0.01, 0.02, 0.1}, Chi in {0.01, 0.02, 0.05, 0.1}.

    Parameters:
        by_area: Use area-weighted stability index if True
        jitter_amount: Amount of jitter to add to points
        datahighlight: What data to highlight:
            - "leadingeight": L1-L8 points in orange with labels (original behavior)
            - "morallyconsistent": Points with is_consistent_discriminating == True in bright magenta, no labels
    """
    epsilon_values = [0.0, 0.01, 0.02, 0.1]
    chi_values = [0.01, 0.02, 0.05, 0.1]

    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'
    tick_labelsize = 10 * 1.8 * 0.5    # 9pt (50% of original)
    data_label_fontsize = 13.5 * 0.5   # 6.75pt (50% of original)

    # Marker sizes scaled by (3.25/10)^2 ≈ 0.1
    marker_size_unlabeled = 6   # Was 25, scaled down
    marker_size_labeled = 10    # Was 40, scaled down

    for chi in chi_values:
        for eps in epsilon_values:
            file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
            try:
                print(f"Loading {file_path}")
                df = pd.read_csv(file_path)
            except FileNotFoundError as e:
                print(f"Skipping chi={chi}, eps={eps}: {e}")
                continue

            merged = merge_info(df)

            # Add jitter
            merged = add_jitter(merged, 'cooperation_index', y_col, amount=jitter_amount)
            x_jittered = 'cooperation_index_jittered'
            y_jittered = f'{y_col}_jittered'

            plt.figure(figsize=(3.25, 3.25))  # Small square for montage
            sns.set_style('whitegrid')

            if datahighlight == "leadingeight":
                # Original behavior: gray unlabeled points, orange L1-L8 points with labels
                unlabeled = merged[merged['label'] == '']
                labeled = merged[merged['label'].str.startswith('L')]

                # Plot unlabeled points with jitter
                sns.scatterplot(data=unlabeled, x=x_jittered, y=y_jittered,
                                marker='o', s=marker_size_unlabeled, color='gray', alpha=0.005, legend=False)

                # Plot labeled points with jitter
                sns.scatterplot(data=labeled, x=x_jittered, y=y_jittered,
                                marker='s', s=marker_size_labeled, color='orange', alpha=0.9, legend=False)

                # Annotate L1-L8 labels
                label_offset_scale = 3  # Scale up offsets for small plots
                for _, row in labeled.iterrows():
                    txt = row['label']
                    dx, dy = label_offsets.get(txt, default_offset)
                    dx *= label_offset_scale
                    dy *= label_offset_scale
                    ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
                    va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
                    plt.text(
                        row[x_jittered] + dx,
                        row[y_jittered] + dy,
                        txt,
                        fontsize=data_label_fontsize,
                        ha=ha,
                        va=va,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
                    )

            elif datahighlight == "morallyconsistent":
                # Highlight morally consistent strategies in bright magenta, no labels
                non_consistent = merged[merged['is_consistent_discriminating'] != True]
                consistent = merged[merged['is_consistent_discriminating'] == True]

                # Plot non-consistent points in gray
                sns.scatterplot(data=non_consistent, x=x_jittered, y=y_jittered,
                                marker='o', s=marker_size_unlabeled, color='gray', alpha=0.005, legend=False)

                # Plot consistent points in bright magenta
                sns.scatterplot(data=consistent, x=x_jittered, y=y_jittered,
                                marker='o', s=marker_size_unlabeled, color='#FF00FF', alpha=0.015, legend=False)

            plt.tick_params(axis='both', labelsize=tick_labelsize)
            plt.xlabel('')
            plt.ylabel('')
            plt.xlim(-0.02, 1.02)
            plt.ylim(-0.02, 1.02)
            plt.tight_layout()

            output_png = f"scatter_plots/scatterplot_small_{datahighlight}_chi_{chi}_epsilon_{eps}.png"
            plt.savefig(output_png, dpi=450)
            print(f"Saved {output_png}")
            plt.close()

def plot_panel_4x4(by_area=False, jitter_amount=JITTER_AMOUNT, datahighlight="leadingeight"):
    """
    Compose 16 scatter plots into a single 4×4 panel figure.
    Rows: chi = [0.01, 0.02, 0.05, 0.1]
    Columns: epsilon = [0.0, 0.01, 0.02, 0.1]

    Parameters:
        by_area: Use area-weighted stability index if True
        jitter_amount: Amount of jitter to add to points
        datahighlight: What data to highlight:
            - "leadingeight": L1-L8 points in orange with labels
            - "morallyconsistent": Consistent discriminators in bright magenta
    """
    chi_values = [0.01, 0.02, 0.05, 0.1]
    epsilon_values = [0.0, 0.01, 0.02, 0.1]

    y_col = 'stability_index_area' if by_area else 'stability_index_uniform'

    marker_size_unlabeled = 6
    marker_size_labeled = 10
    data_label_fontsize = 13.5 * 0.5  # 6.75pt
    label_offset_scale = 3

    fig, axes = plt.subplots(4, 4, figsize=(13, 13), sharex=True, sharey=True)
    sns.set_style('whitegrid')

    for row, chi in enumerate(chi_values):
        for col, eps in enumerate(epsilon_values):
            ax = axes[row, col]

            file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"Skipping chi={chi}, eps={eps}: file not found")
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            merged = merge_info(df)
            merged = add_jitter(merged, 'cooperation_index', y_col, amount=jitter_amount)
            x_jittered = 'cooperation_index_jittered'
            y_jittered = f'{y_col}_jittered'

            if datahighlight == "leadingeight":
                unlabeled = merged[merged['label'] == '']
                labeled = merged[merged['label'].str.startswith('L')]

                ax.scatter(unlabeled[x_jittered], unlabeled[y_jittered],
                           marker='o', s=marker_size_unlabeled, color='gray', alpha=0.005)
                ax.scatter(labeled[x_jittered], labeled[y_jittered],
                           marker='s', s=marker_size_labeled, color='orange', alpha=0.9)

                for _, pt in labeled.iterrows():
                    txt = pt['label']
                    dx, dy = label_offsets.get(txt, default_offset)
                    dx *= label_offset_scale
                    dy *= label_offset_scale
                    ha = 'center' if dx == 0 else ('right' if dx < 0 else 'left')
                    va = 'center' if dy == 0 else ('top' if dy < 0 else 'bottom')
                    ax.text(
                        pt[x_jittered] + dx,
                        pt[y_jittered] + dy,
                        txt,
                        fontsize=data_label_fontsize,
                        ha=ha, va=va,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
                    )

            elif datahighlight == "morallyconsistent":
                non_consistent = merged[merged['is_consistent_discriminating'] != True]
                consistent = merged[merged['is_consistent_discriminating'] == True]

                ax.scatter(non_consistent[x_jittered], non_consistent[y_jittered],
                           marker='o', s=marker_size_unlabeled, color='gray', alpha=0.005)
                ax.scatter(consistent[x_jittered], consistent[y_jittered],
                           marker='o', s=marker_size_unlabeled, color='#FF00FF', alpha=0.015)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(axis='both', labelsize=9)

            # Column headers on top row
            if row == 0:
                ax.set_title(f"$\\epsilon = {eps}$", fontsize=11)

            # Row labels on left column
            if col == 0:
                ax.set_ylabel(f"$\\chi = {chi}$\nstability index", fontsize=10)

            # X-axis label on bottom row only
            if row == 3:
                ax.set_xlabel("cooperation index", fontsize=10)

    plt.tight_layout()
    os.makedirs("scatter_plots", exist_ok=True)
    output_png = f"scatter_plots/scatterplot_panel_4x4_{datahighlight}_area_{by_area}.png"
    plt.savefig(output_png, dpi=450)
    print(f"Saved {output_png}")
    plt.close()


def scatter_disaggregated(by_area=False, jitter_amount=0.005):
    """Produce a 1×3 figure with cooperation_index vs game-specific stability indices (PD, SH, SG),
    highlighting morally consistent discriminators in magenta."""
    file_path = f"{folder}global_complete_chi_0.02_epsilon_0.02.csv"
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    merged = merge_info(df)

    y_columns = ['stability_index_PD', 'stability_index_SH', 'stability_index_SG']
    panel_titles = ['prisoner\'s dilemma', 'stag hunt', 'snowdrift']

    fig, axes = plt.subplots(1, 3, figsize=(9.75, 3.25))
    sns.set_style('whitegrid')

    for ax, y_col, title in zip(axes, y_columns, panel_titles):
        plt.sca(ax)

        jittered = add_jitter(merged, 'cooperation_index', y_col, amount=jitter_amount)
        x_jittered = 'cooperation_index_jittered'
        y_jittered = f'{y_col}_jittered'

        non_consistent = jittered[jittered['is_consistent_discriminating'] != True]
        consistent = jittered[jittered['is_consistent_discriminating'] == True]

        sns.scatterplot(data=non_consistent, x=x_jittered, y=y_jittered,
                        marker='o', s=6, color='gray', alpha=0.005, legend=False, ax=ax)
        sns.scatterplot(data=consistent, x=x_jittered, y=y_jittered,
                        marker='o', s=6, color='#FF00FF', alpha=0.015, legend=False, ax=ax)

        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.set_xlabel('cooperation index', fontsize=10)
        ax.set_ylabel('stability index', fontsize=10) if ax is axes[0] else ax.set_ylabel('')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    output_png = "scatter_plots/scatter_disaggregated_chi_0.02_epsilon_0.02.png"
    os.makedirs("scatter_plots", exist_ok=True)
    plt.savefig(output_png, dpi=450)
    print(f"Saved {output_png}")
    plt.close()


def plot_disaggregated_density(chi=0.02, eps=0.02, jitter_amount=JITTER_AMOUNT,
                               gridsize=80, out_dir='density_scatter'):
    """
    Game-disaggregated companion to plot_single_labeled_density.

    Three side-by-side panels (prisoner's dilemma, stag hunt, snowdrift) share
    the centrepiece visual language: the ~1M strategy cloud as a log-normalized
    fog-slate hexbin, morally consistent discriminators as a crimson scatter
    overlay, and the Leading Eight as chartreuse diamonds. Axis labels are lower
    case and carry the greek symbols (kappa for cooperation, sigma for
    stability). Saves a PNG to out_dir.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    merged = merge_info(df)

    y_columns = ['stability_index_PD', 'stability_index_SH', 'stability_index_SG']
    panel_titles = ['prisoner’s dilemma', 'stag hunt', 'snowdrift']

    # Composite sizing convention (see DISPLAY_DPI note): figure-inches ==
    # display-inches so shared point sizes render at consistent apparent size.
    # Three panels wide, tuned so each square panel roughly matches the 4x4
    # facet cells while leaving room for larger, readable labels.
    target_px = 1200
    fig_w = target_px / DISPLAY_DPI            # ~16.7 in for the row
    panel_in = fig_w / 3
    fig_h = panel_in * 1.12                    # a little headroom for colorbar
    g = panel_in / 6.5                         # design-detail scale vs 6.5 in

    sns.set_style('white')
    prev_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 0.6 * g,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fog_cmap = LinearSegmentedColormap.from_list(
        'fog_slate',
        ['#FFFFFF', '#D9DEE4', '#9FA9B4', '#5C6B7A', '#2C3E50'],
        N=256,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    last_hb = None

    for ax, y_col, title in zip(axes, y_columns, panel_titles):
        # Background cloud only: drop overlay points (consistent
        # discriminators + Leading Eight) so none are plotted twice.
        bg = merged[(merged['is_consistent_discriminating'] != True)
                    & ~merged['label'].astype(str).str.startswith('L')]
        cd = merged[merged['is_consistent_discriminating'] == True].copy()
        leading = merged[merged['label'].astype(str).str.startswith('L')].copy()
        if not cd.empty:
            cd = add_jitter(cd, 'cooperation_index', y_col, amount=jitter_amount)
        if not leading.empty:
            leading = add_jitter(leading, 'cooperation_index', y_col,
                                 amount=jitter_amount)

        ax.grid(True, which='major', color='#EEEEEE', lw=0.4 * g, zorder=0)
        ax.set_axisbelow(True)

        hb = ax.hexbin(
            bg['cooperation_index'].values,
            bg[y_col].values,
            gridsize=gridsize, extent=(0.0, 1.0, 0.0, 1.0),
            cmap=fog_cmap,
            norm=LogNorm(vmin=1, vmax=max(10, len(bg) // 200)),
            mincnt=1, linewidths=0, edgecolors='none', zorder=1,
        )
        last_hb = hb

        if not cd.empty:
            n_cd = len(cd)
            if n_cd > 3000:
                face_alpha, edge_lw = 0.5, 0.0
            else:
                face_alpha, edge_lw = 0.85, 0.25
            ax.scatter(
                cd['cooperation_index_jittered'],
                cd[f'{y_col}_jittered'],
                s=10 * g**2, marker='o',
                facecolor='#C8102E', edgecolor='white',
                linewidth=edge_lw * g, alpha=face_alpha, zorder=3,
            )

        if not leading.empty:
            ax.scatter(
                leading['cooperation_index_jittered'],
                leading[f'{y_col}_jittered'],
                s=22 * g**2, marker='D',
                facecolor='#C6FF00', edgecolor='#1A1A1A',
                linewidth=0.6 * g, zorder=4,
            )

        sns.despine(ax=ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=12 * 1.4, pad=6 * g)
        ax.tick_params(axis='both', labelsize=10 * 1.4,
                       length=3.5 * g, width=0.5 * g)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Shared colorbar along the top of the row.
    if last_hb is not None:
        cb = fig.colorbar(
            last_hb, ax=axes, location='top',
            shrink=0.32, aspect=35, pad=0.02,
        )
        cb.outline.set_linewidth(0.4 * g)
        cb.ax.tick_params(labelsize=9 * 1.1, length=2.5 * g, width=0.4 * g, pad=1.5)
        cb.set_label('systems / hex (log)', fontsize=10 * 1.1, labelpad=3)

    # Figure-wide axis labels: lower case, greek-annotated.
    fig.supxlabel(r'cooperation index ($\kappa$)', fontsize=12 * 1.6)
    fig.supylabel(r'stability index ($\sigma$)', fontsize=12 * 1.6)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(
        out_dir, f"density_disaggregated_chi_{chi}_epsilon_{eps}.png")
    fig.savefig(out_png, dpi=450)
    print(f"Saved {out_png}")
    plt.close(fig)
    mpl.rcParams.update(prev_rc)


def plot_single_game_density(game, chi=0.02, eps=0.02, jitter_amount=JITTER_AMOUNT,
                             gridsize=80, out_dir='density_scatter'):
    """
    Single-game companion to plot_disaggregated_density.

    Renders one panel of the disaggregated row on its own, keeping the same
    visual language (log-normalized fog-slate hexbin, crimson morally
    consistent discriminators, chartreuse Leading Eight diamonds). Because it
    stands alone rather than sharing the panel's colorbar-as-key, it carries a
    compact marker legend in the upper left. `game` is one of 'PD', 'SG', 'SH'.
    Saves a PNG tagged with the lower-case game key.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    game = game.upper()
    game_info = {
        'PD': ('stability_index_PD', 'prisoner’s dilemma'),
        'SH': ('stability_index_SH', 'stag hunt'),
        'SG': ('stability_index_SG', 'snowdrift'),
    }
    if game not in game_info:
        raise ValueError(f"game must be one of {list(game_info)}, got {game!r}")
    y_col, title = game_info[game]

    file_path = f"{folder}global_complete_chi_{chi}_epsilon_{eps}.csv"
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    merged = merge_info(df)

    # Match the per-panel geometry of the disaggregated row so a broken-out
    # figure reads at the same apparent scale as one of its cells.
    target_px = 1200
    panel_in = (target_px / DISPLAY_DPI) / 3
    fig_w = panel_in
    fig_h = panel_in
    g = panel_in / 6.5

    sns.set_style('white')
    prev_rc = mpl.rcParams.copy()
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.linewidth': 0.6 * g,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fog_cmap = LinearSegmentedColormap.from_list(
        'fog_slate',
        ['#FFFFFF', '#D9DEE4', '#9FA9B4', '#5C6B7A', '#2C3E50'],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    # Background cloud only: drop overlay points (consistent discriminators +
    # Leading Eight) so none are plotted twice.
    bg = merged[(merged['is_consistent_discriminating'] != True)
                & ~merged['label'].astype(str).str.startswith('L')]
    cd = merged[merged['is_consistent_discriminating'] == True].copy()
    leading = merged[merged['label'].astype(str).str.startswith('L')].copy()
    if not cd.empty:
        cd = add_jitter(cd, 'cooperation_index', y_col, amount=jitter_amount)
    if not leading.empty:
        leading = add_jitter(leading, 'cooperation_index', y_col,
                             amount=jitter_amount)

    ax.grid(True, which='major', color='#EEEEEE', lw=0.4 * g, zorder=0)
    ax.set_axisbelow(True)

    hb = ax.hexbin(
        bg['cooperation_index'].values,
        bg[y_col].values,
        gridsize=gridsize, extent=(0.0, 1.0, 0.0, 1.0),
        cmap=fog_cmap,
        norm=LogNorm(vmin=1, vmax=max(10, len(bg) // 200)),
        mincnt=1, linewidths=0, edgecolors='none', zorder=1,
    )

    if not cd.empty:
        n_cd = len(cd)
        if n_cd > 3000:
            face_alpha, edge_lw = 0.5, 0.0
        else:
            face_alpha, edge_lw = 0.85, 0.25
        ax.scatter(
            cd['cooperation_index_jittered'],
            cd[f'{y_col}_jittered'],
            s=10 * g**2, marker='o',
            facecolor='#C8102E', edgecolor='white',
            linewidth=edge_lw * g, alpha=face_alpha, zorder=3,
        )

    if not leading.empty:
        ax.scatter(
            leading['cooperation_index_jittered'],
            leading[f'{y_col}_jittered'],
            s=22 * g**2, marker='D',
            facecolor='#C6FF00', edgecolor='#1A1A1A',
            linewidth=0.6 * g, zorder=4,
        )

    sns.despine(ax=ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    # Keep the data box square regardless of the space constrained_layout
    # reserves for the colorbar on the right.
    ax.set_box_aspect(1)
    ax.set_title(title, fontsize=12 * 1.4, pad=6 * g)
    ax.tick_params(axis='both', labelsize=10 * 1.4,
                   length=3.5 * g, width=0.5 * g)
    ax.set_xlabel(r'cooperation index ($\kappa$)', fontsize=12 * 1.4)
    ax.set_ylabel(r'stability index ($\sigma$)', fontsize=12 * 1.4)

    # Inset colorbar (vertical), parked just outside the right edge of the axes
    # so it never collides with the data, running bottom-to-top alongside the
    # plot.
    cax = ax.inset_axes([1.03, 0.325, 0.025, 0.35])
    cb = fig.colorbar(hb, cax=cax, orientation='vertical')
    cb.outline.set_linewidth(0.4 * g)
    cb.ax.tick_params(labelsize=7 * g * 1.1, length=2 * g, width=0.4 * g, pad=1.5 * g)
    cb.set_label('systems / hex (log)', fontsize=7.5 * g * 1.1, labelpad=2 * g)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(
        out_dir,
        f"density_{game.lower()}_chi_{chi}_epsilon_{eps}.png")
    fig.savefig(out_png, dpi=450)
    print(f"Saved {out_png}")
    plt.close(fig)
    mpl.rcParams.update(prev_rc)


if __name__ == "__main__":
    print("Generating plot…")
    # plot_panel_4x4_zoom(by_area=False)
    # plot_panel_4x4_zoom(by_area=True)
    # plot_zoom(by_area=False)
    # plot_zoom(by_area=True)
    # plot_zoom_labeled(by_area=False, generous=True)
    # plot_zoom_labeled(by_area=True)
    # plot_pareto(by_area=False, generous=True)
    # plot_single_labeled_jittered(chi=0.02, eps=0.02, by_area=False)
    # Per-file sweep:
    out_dir = "density_scatter"
    for file in files:
        match = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", file)
        if not match:
            continue
        chi, eps = match.group(1), match.group(2)
        for by_area in (False, True):
            try:
                plot_single_labeled_density(
                    chi=chi, eps=eps, by_area=by_area, out_dir=out_dir,
                )
            except FileNotFoundError as e:
                print(f"Skipping chi={chi}, eps={eps}, area={by_area}: {e}")
        try:
            plot_single_zoom_jittered(chi=chi, eps=eps, by_area=False)
        except FileNotFoundError as e:
            print(f"Skipping zoom chi={chi}, eps={eps}: {e}")

    # 4x4 panel facets, one per stability metric.
    # plot_panel_4x4_density(by_area=False)
    # plot_panel_4x4_density(by_area=True)
    # plot_single_zoom_jittered(chi=0.02, eps=0.02, by_area=False)
    # scatter_disaggregated()
    # plot_panel_4x4(by_area=False, datahighlight="leadingeight")
    # plot_panel_4x4(by_area=False, datahighlight="morallyconsistent")
    print("Done.")