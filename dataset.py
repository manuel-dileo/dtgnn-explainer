from torch_geometric.datasets import BitcoinOTC
from torch_geometric.transforms import Constant

import pandas as pd
import torch
from torch_geometric.data import Data
from datetime import datetime, timedelta

def load_dataset(dataset:str):
    if dataset=='bitcoinotc':
        data = BitcoinOTC('dataset/bitcoinotc')
    elif dataset=='reddit-title':
        data = load_reddit_title()
    elif dataset=='uci-message':
        data = load_college_msg()
    elif dataset=='enron':
        data = load_enron()
    elif dataset=='email-eu':
        data = load_email_eu()
    elif dataset=='askubuntu':
        data = load_askubuntu()
    return data

def load_reddit_title():
    # File path
    file_path = "dataset/roland_public_data/reddit-title.tsv"
    
    # Read the CSV file
    df = pd.read_csv(file_path, sep='\\t', header=0)  # Assuming tab-separated file
    
    # Convert SOURCE_SUBREDDIT and TARGET_SUBREDDIT to integer IDs
    df['SOURCE_ID'] = df['SOURCE_SUBREDDIT'].astype('category').cat.codes
    df['TARGET_ID'] = df['TARGET_SUBREDDIT'].astype('category').cat.codes

    
    # Convert TIMESTAMP to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S', errors='raise')

    
    # Determine the range of weeks
    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()
    
    # Create weekly bins
    weeks = []
    current_time = start_time
    while current_time <= end_time:
        weeks.append((current_time, current_time + timedelta(weeks=1)))
        current_time += timedelta(weeks=1)
    
    # List to store Data objects for each week
    temporal_graphs = []
    
    # Create weekly snapshots
    for start, end in weeks:
        # Filter edges for the current week
        weekly_edges = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)]
    
        # Extract edge indices
        edge_index = torch.tensor(weekly_edges[['SOURCE_ID', 'TARGET_ID']].values.T, dtype=torch.long)
    
        # Extract edge attributes (if needed)
        edge_attr = torch.tensor(weekly_edges['LINK_SENTIMENT'].values, dtype=torch.float).unsqueeze(1)
    
        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
        # Add node properties if needed
        # For example, set x to a tensor of zeros for each node (if no other node features are provided)
        num_nodes = max(edge_index.max().item() + 1, df['SOURCE_ID'].max() + 1, df['TARGET_ID'].max() + 1)
        data.x = torch.randn((num_nodes, 1))  # Example node feature placeholder
    
        temporal_graphs.append(data)
    
    return temporal_graphs

def load_college_msg():
    # File path
    file_path = "dataset/roland_public_data/CollegeMsg.txt"
    
    # Read the space-separated file with no header
    df = pd.read_csv(file_path, sep=' ', header=None, names=['SOURCE', 'TARGET', 'TIMESTAMP'])
    
    # Convert SOURCE and TARGET to integer IDs (if they are not already integers)
    df['SOURCE_ID'] = df['SOURCE'].astype('category').cat.codes
    df['TARGET_ID'] = df['TARGET'].astype('category').cat.codes
    
    # Convert TIMESTAMP to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s', errors='raise')  # Assuming timestamp is in seconds
    
    # Determine the range of weeks
    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()
    
    # Fix: Align end_time to the start of the next week to include all remaining data
    end_time = start_time + timedelta(weeks=((end_time - start_time).days // 7) + 1)
    
    # Create weekly bins
    weeks = []
    current_time = start_time
    while current_time < end_time:  # Ensure no weeks are skipped
        weeks.append((current_time, current_time + timedelta(weeks=1)))
        current_time += timedelta(weeks=1)
    
    # List to store Data objects for each week
    temporal_graphs = []
    
    # Create weekly snapshots
    for start, end in weeks:
        # Filter edges for the current week
        weekly_edges = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)]
        
        if weekly_edges.empty:  # Skip weeks with no data
            continue
    
        # Extract edge indices
        edge_index = torch.tensor(weekly_edges[['SOURCE_ID', 'TARGET_ID']].values.T, dtype=torch.long)
    
        # If there are edge attributes, define them (no additional attributes in this dataset)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float)  # Placeholder attribute (e.g., weight of 1)
    
        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
        # Add node properties if needed
        num_nodes = max(edge_index.max().item() + 1, df['SOURCE_ID'].max() + 1, df['TARGET_ID'].max() + 1)
        data.x = torch.randn((num_nodes, 1))  # Example node feature placeholder
    
        temporal_graphs.append(data)
    
    return temporal_graphs

def load_enron():
    # File path
    file_path = "dataset/enron/out.enron"
    
    # Read the space-separated file, skip the first row
    df = pd.read_csv(file_path, sep=' ', header=None, skiprows=1, usecols=[0, 1, 3], names=['SOURCE', 'TARGET', 'TIMESTAMP'])
    
    # Convert SOURCE and TARGET to integer IDs (if they are not already integers)
    df['SOURCE_ID'] = df['SOURCE'].astype('category').cat.codes
    df['TARGET_ID'] = df['TARGET'].astype('category').cat.codes
    
    # Convert TIMESTAMP to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s', errors='raise')  # Assuming timestamp is in seconds
    
    # Determine the range of weeks
    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()
    
    # Align end_time to the start of the next full week
    end_time = start_time + timedelta(weeks=((end_time - start_time).days // 7) + 1)
    
    # Create weekly bins
    weeks = []
    current_time = start_time
    while current_time < end_time:  # Ensure no weeks are skipped
        weeks.append((current_time, current_time + timedelta(weeks=1)))
        current_time += timedelta(weeks=1)
    
    # List to store Data objects for each week
    temporal_graphs = []
    
    # Create weekly snapshots
    for start, end in weeks:
        # Filter edges for the current week
        weekly_edges = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)]
        
        if weekly_edges.empty:  # Skip weeks with no data
            continue
    
        # Extract edge indices
        edge_index = torch.tensor(weekly_edges[['SOURCE_ID', 'TARGET_ID']].values.T, dtype=torch.long)
    
        # If there are edge attributes, define them (no additional attributes in this dataset)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float)  # Placeholder attribute (e.g., weight of 1)
    
        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
        # Add node properties if needed
        num_nodes = max(edge_index.max().item() + 1, df['SOURCE_ID'].max() + 1, df['TARGET_ID'].max() + 1)
        data.x = torch.randn((num_nodes, 1))  # Example node feature placeholder
    
        temporal_graphs.append(data)
    
    # Output: a list of weekly PyTorch Geometric Data objects
    print(f"Generated {len(temporal_graphs)} weekly snapshots.")
    
    return temporal_graphs

import pandas as pd
import torch
from torch_geometric.data import Data
from datetime import datetime, timedelta

def load_email_eu():
    # File path
    file_path = "dataset/email-eu/email-Eu.txt"
    
    # Read the space-separated file, skip the first row
    df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2], names=['SOURCE', 'TARGET', 'TIMESTAMP'])
    
    # Convert SOURCE and TARGET to integer IDs (if they are not already integers)
    df['SOURCE_ID'] = df['SOURCE'].astype('category').cat.codes
    df['TARGET_ID'] = df['TARGET'].astype('category').cat.codes
    
    # Convert TIMESTAMP to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s', errors='raise')  # Assuming timestamp is in seconds
    
    # Determine the range of days
    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()
    
    # Align end_time to the start of the next full day
    end_time = start_time + timedelta(days=(end_time - start_time).days + 1)
    
    # Create daily bins
    days = []
    current_time = start_time
    while current_time < end_time:  # Ensure no days are skipped
        days.append((current_time, current_time + timedelta(days=1)))
        current_time += timedelta(days=1)
    
    # List to store Data objects for each day
    temporal_graphs = []
    
    # Create daily snapshots
    for start, end in days:
        # Filter edges for the current day
        daily_edges = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)]
        
        if daily_edges.empty:  # Skip days with no data
            continue
    
        # Extract edge indices
        edge_index = torch.tensor(daily_edges[['SOURCE_ID', 'TARGET_ID']].values.T, dtype=torch.long)
    
        # If there are edge attributes, define them (no additional attributes in this dataset)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float)  # Placeholder attribute (e.g., weight of 1)
    
        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr)
    
        # Add node properties if needed
        num_nodes = max(edge_index.max().item() + 1, df['SOURCE_ID'].max() + 1, df['TARGET_ID'].max() + 1)
        data.x = torch.randn((num_nodes, 1))  # Example node feature placeholder
    
        temporal_graphs.append(data)
    
    # Output: a list of daily PyTorch Geometric Data objects
    print(f"Generated {len(temporal_graphs)} daily snapshots.")
    
    return temporal_graphs

import pandas as pd
import torch
from torch_geometric.data import Data
from datetime import timedelta

def load_askubuntu():
    # File path
    file_path = "dataset/askubuntu/sx-askubuntu-a2q.txt"

    # Read the space-separated file with no header
    df = pd.read_csv(file_path, sep=' ', header=None, names=['SOURCE', 'TARGET', 'TIMESTAMP'])

    # Convert SOURCE and TARGET to integer IDs (if they are not already integers)
    df['SOURCE_ID'] = df['SOURCE'].astype('category').cat.codes
    df['TARGET_ID'] = df['TARGET'].astype('category').cat.codes

    # Convert TIMESTAMP to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s', errors='raise')  # Assuming timestamp is in seconds

    # Determine the range of days
    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()

    # Fix: Align end_time to the start of the next day to include all remaining data
    end_time = start_time + timedelta(days=((end_time - start_time).days + 1))

    # Create daily bins
    days = []
    current_time = start_time
    while current_time < end_time:  # Ensure no days are skipped
        days.append((current_time, current_time + timedelta(days=1)))
        current_time += timedelta(days=1)

    # List to store Data objects for each day
    temporal_graphs = []

    # Create daily snapshots
    for start, end in days:
        # Filter edges for the current day
        daily_edges = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < end)]

        if daily_edges.empty:  # Skip days with no data
            continue

        # Extract edge indices
        edge_index = torch.tensor(daily_edges[['SOURCE_ID', 'TARGET_ID']].values.T, dtype=torch.long)

        # If there are edge attributes, define them (no additional attributes in this dataset)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float)  # Placeholder attribute (e.g., weight of 1)

        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Add node properties if needed
        num_nodes = max(edge_index.max().item() + 1, df['SOURCE_ID'].max() + 1, df['TARGET_ID'].max() + 1)
        data.x = torch.randn((num_nodes, 1))  # Example node feature placeholder

        temporal_graphs.append(data)

    print(f"Generated {len(temporal_graphs)} daily snapshots.")
    
    return temporal_graphs