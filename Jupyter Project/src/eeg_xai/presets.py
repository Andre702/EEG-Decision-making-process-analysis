"""Gotowe konfiguracje eksperymentów. Warianty robimy przez dataclasses.replace()."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .config import (
    AnalysisConfig,
    DatasetConfig,
    ExperimentConfig,
    PreprocessConfig,
    RepresentationConfig,
    TrainConfig,
)

PHYSIONET = DatasetConfig("PHYSIONET", "/content/drive/MyDrive/EEG_Physio", sfreq=160)
GIGASCIENCE = DatasetConfig("GIGASCIENCE", "/content/drive/MyDrive/EEG_Giga_New", sfreq=512)

# Zbiór testowy wybrany ręcznie (dotychczasowe "allegedly_best_patients")
BEST_PATIENTS = ("S031", "S041", "S043", "S048", "S063", "S068", "S095")

MOTOR_CHANNELS = (
    "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "FT7", "F7", "F8", "FT8", "T9", "T10",
)  # fmt: skip

_TRAIN = TrainConfig(test_subjects=BEST_PATIENTS)
_ANALYSIS = AnalysisConfig(target_subjects=("S031", "S041", "S043", "S048", "S063", "S068"))


def _experiment(
    feature_method: str,
    *,
    l_freq: float | None,
    h_freq: float | None,
    channels: tuple[str, ...] | None = None,
    datasets: tuple[DatasetConfig, ...] = (PHYSIONET,),
) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=datasets,
        preprocess=PreprocessConfig(l_freq=l_freq, h_freq=h_freq, selected_channels=channels),
        representation=RepresentationConfig(feature_method=feature_method),
        train=_TRAIN,
        analysis=_ANALYSIS,
    )


def raw() -> ExperimentConfig:
    """Surowe dane: bez filtra, wszystkie elektrody."""
    return _experiment("raw", l_freq=None, h_freq=None)


def filtered() -> ExperimentConfig:
    """Filtr pasmowy 6-28 Hz, wszystkie elektrody."""
    return _experiment("raw", l_freq=6, h_freq=28)


def filtered_motor() -> ExperimentConfig:
    """Filtr pasmowy 6-28 Hz i wybrane elektrody okolicy ruchowej."""
    return _experiment("raw", l_freq=6, h_freq=28, channels=MOTOR_CHANNELS)


def tsp() -> ExperimentConfig:
    """Okienkowy Tangent Space Projection na danych filtrowanych 6-28 Hz."""
    return _experiment("tsp", l_freq=6, h_freq=28)


PRESETS: dict[str, Callable[[], ExperimentConfig]] = {
    "raw": raw,
    "filtered": filtered,
    "filtered_motor": filtered_motor,
    "tsp": tsp,
}


def both_datasets(cfg: ExperimentConfig, resample_to: float = 160) -> ExperimentConfig:
    """Zamienia konfigurację jednozbiorową na Physionet + GigaScience (ze wspólnym resamplingiem)."""
    return replace(
        cfg,
        datasets=(replace(PHYSIONET, prefix="PHYS"), replace(GIGASCIENCE, prefix="GIGA")),
        preprocess=replace(cfg.preprocess, resample_to=resample_to),
    )
