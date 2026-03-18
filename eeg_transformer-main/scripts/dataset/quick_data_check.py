import mne
import os
import numpy as np

DATA_PATH = "./preprocessed_data/Physionet/S001/"

# Pierwszy plik .fif
fif_file = [f for f in os.listdir(DATA_PATH) if f.endswith('.fif')][0]
full_path = os.path.join(DATA_PATH, fif_file)

print(f"Sprawdzanie pliku: {fif_file}")

# Wczytywanie epok
epochs = mne.read_epochs(full_path, preload=True, verbose=False)
events = epochs.events[:, -1] # Ostatnia kolumna powinna być ID zdarzenia

# Sprawdzanie klas
unique_events, counts = np.unique(events, return_counts=True)
print(f"\nZnalezione ID zdarzeń: {unique_events}")
print(f"Ilość próbek w każdej klasie: {counts}")

# Sprawdzanie skali
data = epochs.get_data(copy=True)
mean_val = np.mean(data)
std_val = np.std(data)
min_val = np.min(data)
max_val = np.max(data)

print(f"\nSkala danych")
print(f"  Średnia: {mean_val:.10f}")
print(f"  Odchylenie (Std): {std_val:.10f}")
print(f"  Min: {min_val:.10f}")
print(f"  Max: {max_val:.10f}")

if std_val < 0.001:
    print("\nDane mają bardzo małe wartości! Wymagana normalizacja?")