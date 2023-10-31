import itertools
import numpy as np
import random
import math
import pandas as pd
import os 
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


import networkx as nx


def simulate_head(num_neuron, back_firing, peak_firing, tuning_width, data_length):

    hd_sim = np.random.uniform(0, 2 * math.pi, data_length)
    rf_hd = np.random.uniform(0, 2 * math.pi, num_neuron)
    rate_hd = np.zeros((num_neuron, data_length))
    spikes_hd = np.zeros((num_neuron, data_length))

    # Calculate the absolute difference between rf_hd and hd_sim for all neurons and time steps
    distances = np.abs(rf_hd[:, np.newaxis] - hd_sim)

    # Wrap distances greater than pi
    distances = np.minimum(distances, 2 * np.pi - distances)

    # Calculate the squared distances for all neurons and time steps
    distances_squared = distances ** 2

    # Calculate the response for all neurons and time steps using vectorized operations
    response = np.log(back_firing) + (np.log(peak_firing / back_firing)) * np.exp(-distances_squared / (2 * tuning_width))

    # Calculate rate_hd for all neurons and time steps   
    rate_hd = np.exp(response)

    # Generate spikes_hd using vectorized operations
    spikes_hd = np.random.poisson(lam=rate_hd)
    
    df_head = pd.DataFrame(spikes_hd.T)
    column_mapping = {col: f'head_neuron_{col}' for col in df_head.columns}
    # Rename the columns using the mapping
    df_head.rename(columns=column_mapping, inplace=True)
        
    return df_head, hd_sim, rf_hd

def simulate_state(num_neuron, num_states, frequency, data_length):
    
    state_fr = np.repeat(frequency, num_states)

    state_sim = np.random.choice(np.arange(1, num_states + 1), data_length, replace=True)
    state_tuned = np.random.choice(np.arange(1, num_states + 1), num_neuron, replace=True)
    state_mat = np.zeros((num_neuron, num_states))
    state_mat[np.arange(num_neuron), state_tuned - 1] = 1

    rate_mat = state_mat * state_fr

    rate_state = np.zeros((num_neuron, data_length))
    spikes_state = np.zeros((num_neuron, data_length))


    rate_state = rate_mat[:, state_sim - 1]
    spikes_state = np.random.poisson(lam=rate_state)
    
    df_states = pd.DataFrame(spikes_state.T)
    column_mapping = {col: f'state_neuron_{col}' for col in df_states.columns}
    # Rename the columns using the mapping
    df_states.rename(columns=column_mapping, inplace=True)

    return df_states, state_sim, state_tuned


def compute_absolute_correlation(df, target_column):
    """
    Compute the absolute Pearson correlation coefficient between the target column
    and all other columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame
      The DataFrame containing the data.
    - target_column: str
      The name of the target column for correlation measurement.

    Returns:
    - correlation_distances: pandas Series
      A Series containing the absolute correlation distances for each column
      (excluding the target column).
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' not found in the DataFrame.")

    # Calculate the absolute correlation distances
    correlation_distances = df.corr().abs()[target_column].drop(target_column)

    return correlation_distances

def plot_correlation_distances(correlation_distances, target_column):
    """
    Plot the correlation distances between the target column and all other columns
    using the same style as in the provided code.

    Parameters:
    - correlation_distances: pandas Series
      A Series containing the absolute correlation distances for each column.
    - df: pandas DataFrame
      The DataFrame containing the data.
    - target_column: str
      The name of the target column for correlation measurement.
    """
    # Sort the distances in descending order and get the corresponding column names
    sorted_correlation_distances = correlation_distances.sort_values(ascending=False)
    sorted_columns = sorted_correlation_distances.index

    # Define colors for "head" and "state" variables
    colors = []
    for column in sorted_columns:
        if column.startswith("head"):
            colors.append("blue")  # You can choose any color you prefer for "head" variables
        elif column.startswith("state"):
            colors.append("red")  # You can choose any color you prefer for "state" variables
        else:
            colors.append("gray")  # You can set a default color for other variables

    # Plot the sorted absolute correlation distances with different colors
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sorted_columns, sorted_correlation_distances, color=colors)
    plt.xlabel('Columns')
    plt.ylabel('Absolute Correlation Distance')
    plt.title(f'Sorted Absolute Correlation Distance of {target_column} to Other Columns (Descending)')
    plt.xticks(rotation=45)

    # Create custom legends for "head" and "state" variables
    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Head Neurons'),
        Line2D([0], [0], color='red', lw=4, label='State Neurons'),
    ]

    plt.legend(handles=legend_elements)

    plt.show()


def get_sim_id(file):
    file_end = str.split(file,"/")[1]
    file_final = str.split(file_end, ".")[:-1]
    return ".".join(file_final)


def plot_mse_distances(target_column, df):
    # Create a StandardScaler to normalize the DataFrame
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Calculate the mean squared error (MSE) distances from the target column
    mse_distances = []
    for column in df_normalized.columns:
        if column != target_column:
            mse = mean_squared_error(df_normalized[target_column], df_normalized[column])
            mse_distances.append(mse)

    # Sort the distances in ascending order and get the corresponding column names
    sorted_mse_distances, sorted_columns = zip(*sorted(zip(mse_distances, df_normalized.columns)))

    # Define colors for "head" and "state" variables
    colors = []
    for column in sorted_columns:
        if column.startswith("head"):
            colors.append("blue")  # You can choose any color you prefer for "head" variables
        elif column.startswith("state"):
            colors.append("red")  # You can choose any color you prefer for "state" variables
        else:
            colors.append("gray")  # You can set a default color for other variables

    # Plot the sorted mean MSE distances with different colors
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sorted_columns, sorted_mse_distances, color=colors)
    plt.xlabel('Columns')
    plt.ylabel('Mean Squared Error Distance')
    plt.title(f'Sorted MSE Distance of {target_column} to Other Columns (Ascending)')
    plt.xticks(rotation=45)

    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Head Neurons'),
        Line2D([0], [0], color='red', lw=4, label='State Neurons'),
    ]

    plt.legend(handles=legend_elements)

    plt.show()



def plot_transition_graph(transition_matrix, path):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes corresponding to the states
    num_states = transition_matrix.shape[0]
    for i in range(num_states):
        G.add_node(i)

    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i, j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i, j])

    pos = nx.spring_layout(G)  # Layout for positioning nodes
    labels = {i: f'State {i+1}' for i in range(num_states)}  # Node labels

    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, node_color='skyblue',
            font_size=10, font_color='black',
            font_weight='bold', width=[d['weight'] * 4 for (u, v, d) in G.edges(data=True)],
            arrowstyle='-')

    # Save plot
    if path:
        file_path = f'models/{path}/graph.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300)  # Example: PNG format with 300 DPI

    # Display the plot
    plt.show()



def scatterplot_with_color(y_head, states_head, state_labels=None, path=''):
    use_color = True
    if not state_labels:
        use_color = False
        state_labels = np.ones(len(y_head))
    
    data_df = pd.DataFrame({
        'HeadDirection': y_head,
        'StateHead': states_head,
        "StateCategory" : state_labels
    })
    
    # Create custom x-axis labels
    if use_color:
        palette = sns.color_palette("Set1", n_colors=len(set(state_labels)))
        state_labels = [f'State {i}' for i in range(1, len(set(states_head)) + 1)]

    # Create the plot using Seaborn with coloring by 'StateHead'
    plt.figure(figsize=(10, 6))
    if use_color:
        sns.scatterplot(data=data_df, x='StateHead', y='HeadDirection', hue='StateCategory', palette=palette,
                    alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics
    else: 
        sns.scatterplot(data=data_df, x='StateHead', y='HeadDirection',
                    alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics      
    plt.xlabel('State')
    plt.ylabel('Head Direction (Radians)')
    plt.title('Head Direction vs. State Categorization')

    # Save plot
    if path:
        file_path = f'models/{path}/scatterplot.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Example: PNG format with 300 DPI

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_head_direction_vs_state(y_head, states_head, path=''):
    data_df = pd.DataFrame({
        'HeadDirection': y_head,
        'StateCategory': states_head
    })

    # Create custom x-axis labels
    n_states = len(set(states_head))
    state_labels = [f'State {i}' for i in range(1, n_states + 1)]
    palette = sns.color_palette("husl", n_colors=len(set(states_head)))


    # Create the plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_df, x='StateCategory', y='HeadDirection', palette=palette, alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics
    plt.xticks(range(n_states), labels=state_labels, rotation=45, ha='right')  # Customize x-axis labels
    plt.xlabel('State')
    plt.ylabel('Head Direction (Radians)')
    plt.title('Head Direction vs. State Categorization')

    # Save plot
    if path:
        file_path = f'models/{path}/scatterplot.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Example: PNG format with 300 DPI

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_polar_scatter(y_head, states_head, path=''):
    states = states_head
    n_states = len(set(states))
    angle_data = y_head
    t_end = len(states)
    palette = sns.color_palette("husl", n_states)
    legend_labels = [f"State {i+1}" for i in range(n_states)]

    # Create a time vector
    t = np.arange(1, t_end + 1) / t_end + 0.05

    # Create a polar scatter plot
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, polar=True)
    sc = ax.scatter(y_head, t, c=[palette[x] for x in states], s=20, alpha=1)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set x and y labels to blank
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")

    # Create a custom legend outside the polar plot
    legend_labels = [f"State {i+1}" for i in range(n_states)]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markersize=10, markerfacecolor='C{}'.format(i))
                      for i, label in enumerate(legend_labels)]

    # Adjust the legend position using bbox_to_anchor
    plt.legend(handles=legend_handles, loc='upper right', title="States", bbox_to_anchor=(1.2, 1))

    plt.title("Polar Scatter Plot")

    # Save plot
    if path:
        file_path = f'models/{path}/polar.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Example: PNG format with 300 DPI

    # Plot
    plt.show()


def upper_triangular_values(matrix):
    values_dict = {}
    n = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            values_dict[(i, j)] = matrix[i, j]

    sorted_values_dict = dict(sorted(values_dict.items(), key=lambda item: item[1], reverse = False))

    return sorted_values_dict


def correlation_linkage(X):
    corr_matrix =  1 - np.abs(np.corrcoef(X))
    
    sorted_dict = upper_triangular_values(corr_matrix)
    n = X.shape[0]
    num_members = { i :  1 for i in range(n)}
    index_map = {i : i for i in range(n)}
    
    linkage_matrix = np.zeros((n-1, 4), dtype=float)
    idx = n
    for k in range(len(sorted_dict)):
        # Find the next closest pair
        i, j = list(sorted_dict.keys())[k]
        
        # Find which cluster the data points are member of
        i_new = index_map[i]
        j_new = index_map[j]
        
        # If they are member of the same cluster, continue to next itteration
        if i_new == j_new:
            print("cool")
            continue
        
        # Save that they are member of the next cluster that is to be made
        index_map[i] = idx
        index_map[j] = idx
        
        # Save the new cluster and it's children
        num_members[idx] = num_members[i_new] + num_members[j_new]
        
        # Update linkage matrix
        linkage_matrix[idx - n , 0] = i_new
        linkage_matrix[idx - n , 1] = j_new
        linkage_matrix[idx - n , 2] = list(sorted_dict.values())[k]
        linkage_matrix[idx - n , 3] = num_members[idx]
        
        # Update index
        idx += 1
    
    return linkage_matrix