"""Wykresy statyczne: mapy cieplne (kanały x czas), słupki ważności, profile elektrod."""
import matplotlib.pyplot as plt
import numpy as np

# (...) importy dopasujemy po wklejeniu ciał funkcji


def plot_global_heatmap_and_bars(
    heatmap_data,
    analysis_results,
    title_suffix="All Attributions",
    channel_names=None,
    save_path=None,
    show_plot=True,
):
    """
    Plots an aggregated heatmap from all the samples as well as a horizontal bar chart
    representing total absolute attention per channel without time parameter.
    """
    ...  # (...)


def plot_difference_heatmap_and_bars(
    heatmap_diff,
    analysis_results,
    title_suffix="Right vs Left Diff",
    channel_names=None,
    save_path=None,
    show_plot=True,
):
    """
    Plots an aggregated heatmap representing difference between 2 classifications attention gradients
    Positive values (Right Hand) are Blue.
    Negative values (Left Hand) are Red.
    The bar chart displays the absolute net difference colored appropriately.
    """
    ...  # (...)


def plot_top_n_heatmap_and_bars(
    heatmap_data,
    analysis_results,
    original_channel_names,
    top_n=15,
    title_suffix="Global Average",
    save_path=None,
    tmin=0.0,
    show_plot=False,
):
    """
    Wykres TYPU 1: Prezentuje N-najbardziej wpływowych par/cech TSP posortowanych od najlepszej.
    """
    ...  # (...)


def plot_type_2_electrode_profiles(
    heatmap_data,
    analysis_results,
    original_channel_names,
    top_n_electrodes=5,
    max_top_pairs_per_plot=15,
    title_suffix="Global",
    save_dir=None,
    show_plot=True,
):
    """
    Wykresy TYPU 2: Tworzy serię oddzielnych wykresów dla N-najbardziej wpływowych elektrod.
    """
    ...  # (...)
