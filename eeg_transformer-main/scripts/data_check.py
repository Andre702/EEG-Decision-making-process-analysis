import mne
import os
import numpy as np

# Ścieżka do Twoich danych (z logów widzę taką):
DATA_PATH = "./preprocessed_data/Physionet/S001/"

# Znajdźmy pierwszy plik .fif
fif_file = [f for f in os.listdir(DATA_PATH) if f.endswith('.fif')][0]
full_path = os.path.join(DATA_PATH, fif_file)

print(f"--- ANALIZA PLIKU: {fif_file} ---")

# 1. Wczytaj epoki
epochs = mne.read_epochs(full_path, preload=True, verbose=False)
events = epochs.events[:, -1] # Ostatnia kolumna to ID zdarzenia

# 2. Sprawdź klasy (Event ID)
unique_events, counts = np.unique(events, return_counts=True)
print(f"\n[KLASY] Znalezione ID zdarzeń: {unique_events}")
print(f"[LICZNOŚĆ] Ile próbek w każdej klasie: {counts}")

# Wyjaśnienie kodów BCI IV 2a (standard GDF):
# 769 (lub 1 w MNE): Lewa ręka
# 770 (lub 2 w MNE): Prawa ręka
# 771 (lub 3 w MNE): Stopy
# 772 (lub 4 w MNE): Język
# Inne kody (np. 1023) to odrzucone próbki.

# 3. Sprawdź skalę danych (Normalization check)
data = epochs.get_data(copy=True)
mean_val = np.mean(data)
std_val = np.std(data)
min_val = np.min(data)
max_val = np.max(data)

print(f"\n[SKALA DANYCH] Statystyki sygnału:")
print(f"  Średnia: {mean_val:.10f}")
print(f"  Odchylenie (Std): {std_val:.10f}")
print(f"  Min: {min_val:.10f}")
print(f"  Max: {max_val:.10f}")

if std_val < 0.001:
    print("\n[ALARM] Dane mają bardzo małe wartości (mikrowolty?)!")
    print("        Transformery wymagają normalizacji (np. Z-score).")
    print("        Wagi modelu są za duże dla tak małych danych.")
else:
    print("\n[OK] Dane wydają się być znormalizowane.")