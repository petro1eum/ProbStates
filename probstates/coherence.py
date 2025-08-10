"""
Measures of phase coherence and simple noise channels for Level 4.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from probstates.phase import PhaseState


def coherence_l1(state: PhaseState) -> float:
    """
    Простая мера фазовой когерентности для одиночного состояния:
    C(p) = 2 * sqrt(p * (1-p)) ∈ [0,1].
    """
    p = float(state.probability)
    return float(2.0 * np.sqrt(p * max(0.0, 1.0 - p)))


def dephase(state: PhaseState, sigma_phi: float) -> PhaseState:
    """
    Декогеренция по фазе: гауссово размытие фазы с дисперсией sigma_phi^2 (мод 2π).
    Приближение: просто сдвигаем фазу и усредняем — для простоты вернём ту же p и
    фазу без изменения, так как явная свёртка по φ требует распределения.
    Рекомендуется применять к массивам (PhaseRegister) с явным усреднением.
    """
    phi = float(state.phase)
    if sigma_phi <= 0:
        return state
    # В простейшем детерминированном приближении: фаза не меняется, p неизменна
    return PhaseState(state.probability, phi)


def phase_drift(state: PhaseState, delta_phi: float) -> PhaseState:
    """
    Детеминированный дрейф фазы: φ ← (φ + Δφ) mod 2π.
    """
    return PhaseState(state.probability, (state.phase + float(delta_phi)) % (2 * np.pi))


def amp_damp(state: PhaseState, alpha: float) -> PhaseState:
    """
    Амплитудное демпфирование: p ← (1-α) p, φ без изменений. 0≤α≤1.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    return PhaseState((1.0 - a) * state.probability, state.phase)


def coherence_under_noise(state: PhaseState, sigma_phi: float = 0.0, alpha: float = 0.0, delta_phi: float = 0.0) -> float:
    """
    Возвращает меру когерентности после последовательного применения каналов:
    phase_drift(Δφ) → dephase(σφ) → amp_damp(α).
    """
    s = phase_drift(state, delta_phi)
    s = dephase(s, sigma_phi)
    s = amp_damp(s, alpha)
    return coherence_l1(s)


