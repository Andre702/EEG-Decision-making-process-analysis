"""Układ elektrod EEGLAB (64 kanały) i rysowanie obrysu głowy do topoplotów."""

EEGLAB_64_COORDS = {
    # Położenia na obrazku elektrod do narysowania
}


def get_channel_coords(channel_names):
    """
    Mapuje nazwy kanałów (niezależnie od tego czy MNE dopisało do nich wielkie/małe
    litery lub słowo 'EEG') na idealną siatkę EEGLAB.
    """
    ...  # (...)


def draw_head_outline(ax):
    """Rysuje realistyczny obrys głowy (Nosek, Uszy) o r=1.0"""
    ...  # (...)
