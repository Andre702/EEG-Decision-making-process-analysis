"""Drobne narzędzia niezależne od konfiguracji."""


def get_device():
    """Zwraca cuda, jeśli dostępna, w przeciwnym razie cpu (torch importowany dopiero tutaj)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
