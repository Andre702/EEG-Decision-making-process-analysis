"""Konfiguracja eksperymentu.

Ten moduł celowo nie importuje torch, mne ani numpy: ładuje się natychmiast, a edytor
i linter nie potrzebują ciężkich zależności, żeby go sprawdzić.

Zasada: jeden obiekt `ExperimentConfig` opisuje całe uruchomienie. Wszystkie ścieżki wyjściowe
(checkpointy, wykresy, wyniki Captum) są wyprowadzane z ustawień eksperymentu, więc różne
konfiguracje nigdy nie nadpisują sobie wyników ani nie wznawiają cudzych checkpointów.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FEATURE_METHODS = ("raw", "tsp")
DATASET_NAMES = ("PHYSIONET", "GIGASCIENCE")


def _fmt(x: float | None) -> str:
    return "none" if x is None else f"{x:g}"


# --------------------------------------------------------------------------------------
# Sekcje konfiguracji
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """Jeden zbiór danych."""

    name: str  # "PHYSIONET" lub "GIGASCIENCE"
    data_path: str  # surowe dane (Dysk Google)
    sfreq: float  # oryginalna częstotliwość próbkowania [Hz]
    prefix: str = ""  # prefiks ID pacjentów; wymagany przy łączeniu zbiorów

    def __post_init__(self) -> None:
        if self.name.upper() not in DATASET_NAMES:
            raise ValueError(f"Nieznany zbiór {self.name!r}. Dozwolone: {DATASET_NAMES}.")
        if self.sfreq <= 0:
            raise ValueError("sfreq musi być dodatnie.")


@dataclass(frozen=True)
class PreprocessConfig:
    """Co robimy z sygnałem przed uczeniem: segment, filtr pasmowy, resampling, elektrody.

    None w l_freq/h_freq/resample_to/selected_channels oznacza: ten krok jest pominięty.
    """

    segment_type: str = "4s"  # część nazwy plików *-epo.fif
    tmin: float = 0.0
    tmax: float = 4.0
    l_freq: float | None = 6
    h_freq: float | None = 28
    resample_to: float | None = None
    selected_channels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tmax <= self.tmin:
            raise ValueError(f"tmax ({self.tmax}) musi być większe niż tmin ({self.tmin}).")
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError(f"l_freq ({self.l_freq}) musi być mniejsze niż h_freq ({self.h_freq}).")
        if self.selected_channels is not None and len(self.selected_channels) == 0:
            raise ValueError("selected_channels jest puste. Użyj None, żeby zostawić wszystkie elektrody.")

    @property
    def segment_seconds(self) -> float:
        return self.tmax - self.tmin

    @property
    def tag(self) -> str:
        """Krótki opis przetwarzania. Używany w nazwach katalogów."""
        if self.l_freq is None and self.h_freq is None:
            band = "nofilt"
        else:
            band = f"bp{_fmt(self.l_freq)}-{_fmt(self.h_freq)}"
        if self.selected_channels is None:
            chans = "allch"
        else:
            digest = hashlib.md5(",".join(sorted(self.selected_channels)).encode()).hexdigest()[:6]
            chans = f"ch{len(self.selected_channels)}-{digest}"
        resample = "" if self.resample_to is None else f"_rs{_fmt(self.resample_to)}"
        return f"{self.segment_type}_{band}_{chans}{resample}"


@dataclass(frozen=True)
class RepresentationConfig:
    """Jak sygnał trafia do modelu: surowy czy okienkowy Tangent Space (TSP)."""

    feature_method: str = "raw"  # "raw" lub "tsp"
    temporal_domain: bool = True  # True: (Batch, Time, Channels); False: (Batch, Channels, Time)
    tsp_window_sec: float = 0.25
    tsp_step_sec: float = 0.10

    def __post_init__(self) -> None:
        if self.feature_method not in FEATURE_METHODS:
            raise ValueError(f"feature_method={self.feature_method!r}. Dozwolone: {FEATURE_METHODS}.")
        if self.tsp_window_sec <= 0 or self.tsp_step_sec <= 0:
            raise ValueError("tsp_window_sec i tsp_step_sec muszą być dodatnie.")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_head: int = 4
    noise_std: float = 0.01  # szum gaussowski stosowany tylko w treningu

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) musi być podzielne przez n_head ({self.n_head}).")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 30
    patience: int = 9
    test_patient_count: int = 7  # losowy hold-out, gdy test_subjects jest None
    test_subjects: tuple[str, ...] | None = None  # jeśli podane: ma pierwszeństwo
    train_model: bool = True  # False: tylko wczytaj najlepszy zapisany checkpoint
    explicit_model_path: str | None = None  # jeśli istnieje: pomija trening i szukanie checkpointu


@dataclass(frozen=True)
class AnalysisConfig:
    max_samples: int = 9999  # limit próbek dla Captum
    plot_rl_separately: bool = True
    generate_type_2_plots: bool = True
    generate_type_3_gifs: bool = True
    target_subjects: tuple[str, ...] = ()  # pacjenci do osobnych analiz
    top_n_features: int = 32
    top_n_electrodes: int = 8
    top_n_max_pairs_per_electrode: int = 32


@dataclass(frozen=True)
class PathsConfig:
    output_root: str = "/content/drive/MyDrive/EEG_results"  # trwałe wyniki (Dysk Google)
    work_dir: str = "./work"  # tymczasowe pliki lokalne (preprocessed .fif)


# --------------------------------------------------------------------------------------
# Eksperyment
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[DatasetConfig, ...]
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    name: str | None = None  # None: nazwa generowana z ustawień

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("Podaj co najmniej jeden zbiór danych.")

        names = [d.name.upper() for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError(f"Powtórzony zbiór danych: {names}.")

        if len(self.datasets) > 1:
            if any(not d.prefix for d in self.datasets):
                raise ValueError("Przy łączeniu zbiorów każdy musi mieć niepusty prefix.")
            prefixes = [d.prefix for d in self.datasets]
            if len(set(prefixes)) != len(prefixes):
                raise ValueError(f"Prefiksy zbiorów muszą być różne: {prefixes}.")
            if len({d.sfreq for d in self.datasets}) > 1 and self.preprocess.resample_to is None:
                raise ValueError(
                    "Zbiory mają różne częstotliwości próbkowania. "
                    "Ustaw preprocess.resample_to na wspólną wartość."
                )

        rep, pre = self.representation, self.preprocess
        if rep.feature_method == "tsp" and rep.tsp_window_sec >= pre.segment_seconds:
            raise ValueError(
                f"Okno TSP ({rep.tsp_window_sec}s) musi być krótsze niż segment ({pre.segment_seconds}s)."
            )

    # --- wartości wyprowadzone
    @property
    def sfreq(self) -> float:
        """Częstotliwość próbkowania danych wchodzących do modelu."""
        if self.preprocess.resample_to is not None:
            return self.preprocess.resample_to
        return self.datasets[0].sfreq

    @property
    def sequence_first(self) -> bool:
        return self.representation.temporal_domain

    @property
    def experiment_name(self) -> str:
        if self.name:
            return self.name
        ds = "+".join(sorted(d.name.lower() for d in self.datasets))
        rep, m, t = self.representation, self.model, self.train
        parts = [ds, rep.feature_method, self.preprocess.tag]
        if rep.feature_method == "tsp":
            parts.append(f"tsp{_fmt(rep.tsp_window_sec)}-{_fmt(rep.tsp_step_sec)}")
        if not rep.temporal_domain:
            parts.append("spatial")
        parts.append(f"bs{t.batch_size}_d{m.d_model}_h{m.n_head}_lr{_fmt(t.lr)}")
        return "_".join(parts)

    # --- ścieżki
    @property
    def run_dir(self) -> Path:
        return Path(self.paths.output_root) / self.experiment_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def global_test_dir(self) -> Path:
        return self.run_dir / "global_test_set"

    @property
    def captum_path(self) -> Path:
        return self.run_dir / f"captum_results_{self.preprocess.segment_type}.joblib"

    def preprocessed_dir(self, dataset: DatasetConfig) -> Path:
        """Preprocessed .fif zależą tylko od zbioru i ustawień przetwarzania (nie od modelu)."""
        return Path(self.paths.work_dir) / "preprocessed" / dataset.name.lower() / self.preprocess.tag

    # --- pomocnicze
    def save(self) -> Path:
        """Zapisuje pełną konfigurację obok wyników (ślad do pracy: co dokładnie uruchomiono)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def describe(self) -> str:
        p, r, m, t = self.preprocess, self.representation, self.model, self.train
        chans = "wszystkie" if p.selected_channels is None else f"{len(p.selected_channels)} wybranych"
        band = "brak" if p.l_freq is None and p.h_freq is None else f"{_fmt(p.l_freq)}-{_fmt(p.h_freq)} Hz"
        lines = [
            f"Eksperyment : {self.experiment_name}",
            f"Zbiory      : {', '.join(d.name for d in self.datasets)} (sfreq={self.sfreq:g} Hz)",
            f"Segment     : {p.segment_type} ({p.tmin:g}-{p.tmax:g} s)",
            f"Filtr       : {band}, elektrody: {chans}",
            f"Reprezent.  : {r.feature_method}, temporal_domain={r.temporal_domain}",
            f"Model       : d_model={m.d_model}, n_head={m.n_head}, noise={m.noise_std:g}",
            f"Trening     : bs={t.batch_size}, lr={t.lr:g}, epoki={t.epochs}, patience={t.patience}",
            f"Wyniki      : {self.run_dir}",
        ]
        if r.feature_method == "tsp":
            lines.insert(5, f"TSP         : okno={r.tsp_window_sec:g}s, krok={r.tsp_step_sec:g}s")
        return "\n".join(lines)
