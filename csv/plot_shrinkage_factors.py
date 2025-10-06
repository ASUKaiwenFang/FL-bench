"""
Shrinkage Factor Visualization Tool for DP-ScaffStein

This script visualizes shrinkage factors from DP-ScaffStein experiments:
- Algorithm 1 (last_noise_server_jse): Line plot with server shrinkage factors
- Algorithm 2/3 (step_noise_step_jse/step_noise_final_jse): Tube plot with min/max/average

Usage:
    python plot_shrinkage_factors.py
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_shrinkage_data(json_path: str) -> Tuple[str, Dict]:
    """
    Load shrinkage factor data from JSON file.

    Args:
        json_path: Path to shrinkage_factors.json file

    Returns:
        Tuple of (algorithm_variant, data_dict)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    algorithm_variant = data.get("algorithm_variant", "unknown")
    shrinkage_data = data.get("data", {})

    return algorithm_variant, shrinkage_data


def plot_algorithm1(data: Dict, output_path: str, title: str = "Algorithm 1: Server-side JSE Shrinkage Factors"):
    """
    Plot Algorithm 1 shrinkage factors (server-side JSE).

    Creates a simple line plot with global epochs on x-axis and shrinkage factors on y-axis.

    Args:
        data: Dictionary mapping epoch to {"server": shrinkage_factor}
        output_path: Output PNG file path
        title: Plot title
    """
    # Extract epochs and shrinkage factors
    epochs = []
    shrinkage_factors = []

    for epoch_str, epoch_data in sorted(data.items(), key=lambda x: int(x[0])):
        epochs.append(int(epoch_str))
        shrinkage_factors.append(epoch_data.get("server", 1.0))

    if not epochs:
        print("No data to plot for Algorithm 1")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line
    ax.plot(epochs, shrinkage_factors,
            linewidth=2.5,
            marker='o',
            markersize=5,
            color='#2E86AB',
            label='Server Shrinkage Factor')

    # Set labels and title
    ax.set_xlabel('Global Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shrinkage Multiplier', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set y-axis limits
    ax.set_ylim([0, 1.05])

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Algorithm 1 plot saved to: {output_path}")
    plt.close()


def plot_algorithm2_3(data: Dict, output_path: str, title: str = "Algorithm 2/3: Client-side JSE Shrinkage Factors"):
    """
    Plot Algorithm 2/3 shrinkage factors with min/max/average visualization.

    Creates a tube plot showing the range (min-max) and average of client shrinkage factors.

    Args:
        data: Dictionary mapping epoch to {client_id: shrinkage_factor}
        output_path: Output PNG file path
        title: Plot title
    """
    # Extract epochs and compute statistics
    epochs = []
    min_values = []
    max_values = []
    avg_values = []

    for epoch_str, epoch_data in sorted(data.items(), key=lambda x: int(x[0])):
        epoch = int(epoch_str)

        # Extract shrinkage factors from all clients
        client_factors = [v for k, v in epoch_data.items() if k.startswith("client_")]

        if not client_factors:
            continue

        epochs.append(epoch)
        min_values.append(np.min(client_factors))
        max_values.append(np.max(client_factors))
        avg_values.append(np.mean(client_factors))

    if not epochs:
        print("No data to plot for Algorithm 2/3")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot filled area (tube) for min-max range
    ax.fill_between(epochs, min_values, max_values,
                     alpha=0.3,
                     color='#A23B72',
                     label='Min-Max Range')

    # Plot average line
    ax.plot(epochs, avg_values,
            linewidth=2.5,
            marker='o',
            markersize=5,
            color='#F18F01',
            label='Average')

    # Plot min and max lines
    ax.plot(epochs, min_values,
            linewidth=1.5,
            linestyle='--',
            color='#A23B72',
            alpha=0.7,
            label='Minimum')

    ax.plot(epochs, max_values,
            linewidth=1.5,
            linestyle='--',
            color='#A23B72',
            alpha=0.7,
            label='Maximum')

    # Set labels and title
    ax.set_xlabel('Global Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shrinkage Multiplier', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set y-axis limits
    ax.set_ylim([0, 1.05])

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Algorithm 2/3 plot saved to: {output_path}")
    plt.close()


def get_available_json_files() -> List[str]:
    """
    Search for shrinkage_factors.json files in the output directory.

    Returns:
        List of paths to shrinkage_factors.json files
    """
    # Search in out/dp_scaffstein directory
    search_pattern = str(Path(__file__).parent.parent / "out" / "dp_scaffstein" / "**" / "shrinkage_factors.json")
    json_files = glob.glob(search_pattern, recursive=True)

    return sorted(json_files)


def interactive_plot():
    """
    Interactive mode to select JSON files and generate plots.
    """
    json_files = get_available_json_files()

    if not json_files:
        print("No shrinkage_factors.json files found in out/dp_scaffstein/")
        print("Please run a DP-ScaffStein experiment first.")
        return

    # Display available files
    print("\n" + "="*80)
    print("Available shrinkage factor data files:")
    print("="*80)
    for idx, filepath in enumerate(json_files, 1):
        rel_path = Path(filepath).relative_to(Path(__file__).parent.parent)
        print(f"[{idx}] {rel_path}")
    print("="*80)

    # Get user selection
    selection = input("\nEnter file number to visualize: ").strip()

    try:
        file_idx = int(selection) - 1
        if file_idx < 0 or file_idx >= len(json_files):
            print("Error: Invalid file number.")
            return
    except ValueError:
        print("Error: Please enter a valid number.")
        return

    selected_file = json_files[file_idx]

    # Load data
    print(f"\nLoading: {selected_file}")
    algorithm_variant, data = load_shrinkage_data(selected_file)
    print(f"Algorithm variant: {algorithm_variant}")
    print(f"Number of epochs: {len(data)}")

    # Determine output path
    output_dir = Path(selected_file).parent
    output_filename = f"shrinkage_visualization_{algorithm_variant}.png"
    output_path = output_dir / output_filename

    # Generate plot based on algorithm variant
    if algorithm_variant == "last_noise_server_jse":
        plot_algorithm1(data, str(output_path))
    elif algorithm_variant in ["step_noise_step_jse", "step_noise_final_jse"]:
        title = f"Algorithm {'2' if algorithm_variant == 'step_noise_step_jse' else '3'}: Client-side JSE Shrinkage Factors"
        plot_algorithm2_3(data, str(output_path), title=title)
    else:
        print(f"Unknown algorithm variant: {algorithm_variant}")
        return

    print(f"\n✓ Visualization complete!")


def main():
    """Main entry point."""
    print("="*80)
    print("DP-ScaffStein Shrinkage Factor Visualization Tool")
    print("="*80)

    interactive_plot()


if __name__ == '__main__':
    main()
