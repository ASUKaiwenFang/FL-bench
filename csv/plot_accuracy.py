"""
Test Accuracy Plotting Tool for FL-bench

This script provides functions to plot and compare test accuracy curves from CSV files.

Usage:
    1. Interactive mode (recommended):
        python plot_accuracy.py

    2. Programmatic usage:
        from plot_accuracy import plot_test_accuracy
        plot_test_accuracy(['file1.csv', 'file2.csv'], output_name='my_plot.png')

Example:
    python plot_accuracy.py
    # Then select files interactively
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_available_csv_files():
    """
    Get list of all available CSV files in the csv folder.

    Returns:
        List[str]: List of CSV filenames (without path)
    """
    csv_dir = Path(__file__).parent
    csv_files = glob.glob(str(csv_dir / "*.csv"))
    csv_files = [os.path.basename(f) for f in csv_files]
    return sorted(csv_files)


def plot_test_accuracy(csv_files, output_name='test_accuracy_comparison.png', title='Test Accuracy Comparison', figsize=(12, 8)):
    """
    Plot test accuracy comparison from multiple CSV files.

    Args:
        csv_files: List of CSV filenames to plot
        output_name: Output PNG filename (default: 'test_accuracy_comparison.png')
        title: Plot title (default: 'Test Accuracy Comparison')
        figsize: Figure size tuple (default: (12, 8))
    """
    csv_dir = Path(__file__).parent

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each CSV file
    for idx, csv_file in enumerate(csv_files):
        csv_path = csv_dir / csv_file

        # Read CSV
        df = pd.read_csv(csv_path)

        # Extract Step and Value columns
        steps = df['Step']
        values = df['Value']

        # Use filename without .csv extension as legend label
        exp_name = csv_file.replace('.csv', '')

        # Plot with different color
        ax.plot(steps, values,
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.8,
                color=colors[idx % len(colors)],
                label=exp_name)

    # Set labels
    ax.set_xlabel('Global Round', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(loc='best', fontsize=20, framealpha=0.9)

    # Add grid
    ax.grid(alpha=0.3, linestyle='--')

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = csv_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved successfully to: {output_path}")

    # Show plot
    plt.show()


def interactive_plot():
    """
    Interactive mode to select CSV files and plot.
    """
    # Get available files
    csv_files = get_available_csv_files()

    if not csv_files:
        print("No CSV files found in the csv folder.")
        return

    # Display available files
    print("\n" + "="*60)
    print("Available CSV files:")
    print("="*60)
    for idx, filename in enumerate(csv_files, 1):
        print(f"[{idx}] {filename}")
    print("="*60)

    # Get user selection
    selection = input("\nEnter file numbers to plot (e.g., 1,2,3) or 'all': ").strip()

    # Parse selection
    if selection.lower() == 'all':
        selected_files = csv_files
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            # Validate indices
            if any(i < 1 or i > len(csv_files) for i in indices):
                print("Error: Invalid file number(s).")
                return
            selected_files = [csv_files[i-1] for i in indices]
        except ValueError:
            print("Error: Invalid input format.")
            return

    # Get output filename
    output_name = input("\nEnter output filename (default: test_accuracy_comparison.png): ").strip()
    if not output_name:
        output_name = 'test_accuracy_comparison.png'

    # Ensure .png extension
    if not output_name.endswith('.png'):
        output_name += '.png'

    # Plot
    print(f"\nPlotting {len(selected_files)} file(s)...")
    plot_test_accuracy(selected_files, output_name=output_name)


if __name__ == '__main__':
    interactive_plot()
