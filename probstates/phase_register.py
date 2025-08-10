"""
probstates/phase_register.py

Фазовый регистр (уровень 4) как массив фазовых состояний размерности 2^n
с поддержкой фазового оракула и преобразования, аналогичного Адамару (FWHT).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Union


def _fwht_inplace(a: np.ndarray) -> None:
    """
    Выполняет по месту дискретное преобразование Уолша–Адамара (FWHT)
    без нормировки. Поддерживает комплексные массивы длины N=2^n.
    """
    n = a.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("FWHT requires length to be a power of two")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def _int_to_bits(x: int, n: int) -> np.ndarray:
    return ((x >> np.arange(n)) & 1).astype(np.uint8)


class PhaseRegister:
    """
    Фазовый регистр из 2^n компонент с комплексными амплитудами α_x.
    α_x = sqrt(p_x) * exp(i φ_x).
    """

    def __init__(self, amplitudes: np.ndarray):
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes must be 1-D array")
        n = amplitudes.shape[0]
        if n & (n - 1) != 0:
            raise ValueError("length must be a power of two (2^n)")
        self._amplitudes = amplitudes.astype(complex, copy=True)

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amplitudes.copy()

    @property
    def num_qubits(self) -> int:
        return int(np.log2(self._amplitudes.shape[0]))

    @classmethod
    def uniform(cls, num_qubits: int) -> "PhaseRegister":
        """
        Создает равномерное состояние: α_x = 1/√(2^n).
        """
        N = 1 << num_qubits
        amp = np.ones(N, dtype=complex) / np.sqrt(N)
        return cls(amp)

    @classmethod
    def from_prob_and_phase(
        cls, probabilities: Sequence[float], phases: Sequence[float]
    ) -> "PhaseRegister":
        if len(probabilities) != len(phases):
            raise ValueError("probabilities and phases must have same length")
        probs = np.asarray(probabilities, dtype=float)
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probabilities must be in [0,1]")
        phases_arr = np.asarray(phases, dtype=float)
        amp = np.sqrt(probs) * np.exp(1j * phases_arr)
        return cls(amp)

    def to_prob_and_phase(self) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.abs(self._amplitudes) ** 2
        phases = np.angle(self._amplitudes)
        return probs, phases

    def normalize(self) -> None:
        norm = np.linalg.norm(self._amplitudes)
        if norm == 0:
            raise ValueError("cannot normalize zero vector")
        self._amplitudes /= norm

    def apply_oracle(self, oracle: Callable[[int], int]) -> None:
        """
        Фазовый оракул: α_x ← α_x * (-1)^{f(x)} = α_x * exp(i π f(x)).

        Args:
            oracle: функция f(x)∈{0,1} над индексом x∈[0, 2^n).
        """
        N = self._amplitudes.shape[0]
        signs = np.fromiter((oracle(x) & 1 for x in range(N)), count=N, dtype=int)
        self._amplitudes *= np.exp(1j * np.pi * signs)

    def hadamard_all(self) -> None:
        """
        Применяет H^{⊗n}: реализация через FWHT и нормировку 1/√N.
        """
        _fwht_inplace(self._amplitudes)
        self._amplitudes /= np.sqrt(self._amplitudes.shape[0])

    def measure_probability(self, index: int) -> float:
        N = self._amplitudes.shape[0]
        if not (0 <= index < N):
            raise ValueError("index out of range")
        return float(np.abs(self._amplitudes[index]) ** 2)

    def argmax_probability(self) -> int:
        return int(np.argmax(np.abs(self._amplitudes) ** 2))

    # --- Тензоризация и частичные операции ---

    def tensor(self, other: "PhaseRegister") -> "PhaseRegister":
        """
        Тензорное произведение регистров: амплитуды по Кронекеру.
        Возвращает новый регистр.
        """
        a = self._amplitudes
        b = other._amplitudes
        out = np.kron(a, b)
        return PhaseRegister(out)

    @classmethod
    def from_kets(cls, *registers: "PhaseRegister") -> "PhaseRegister":
        """Тензоризует несколько регистров слева направо."""
        if len(registers) == 0:
            raise ValueError("need at least one register")
        acc = registers[0]
        for r in registers[1:]:
            acc = acc.tensor(r)
        return acc

    def partial_measure(self, qubit: int) -> Tuple[float, float]:
        """
        Измеряет маргинальные вероятности выбранного кубита (0-базовый, старший слева).
        Возвращает (p0, p1).
        """
        n = self.num_qubits
        if not (0 <= qubit < n):
            raise ValueError("qubit index out of range")
        N = 1 << n
        probs = np.abs(self._amplitudes) ** 2
        p0 = 0.0
        p1 = 0.0
        for x in range(N):
            bit = (x >> (n - 1 - qubit)) & 1
            if bit == 0:
                p0 += probs[x]
            else:
                p1 += probs[x]
        return float(p0), float(p1)

    def povm_measure(self, effects: Sequence[Union[np.ndarray, Sequence[float]]]) -> Tuple[np.ndarray, Tuple["PhaseRegister", ...]]:
        """
        Общая POVM-измерение: набор эффектов E_k (матрицы NxN, положительные, \sum_k E_k = I).
        Допускаются диагональные эффекты в виде 1D-векторов длины N (интерпретируются как diag).

        Возвращает:
            probs (shape K), пост-измерительные нормированные регистры (длины K).
        """
        psi = self._amplitudes
        N = psi.shape[0]
        mats = []
        for E in effects:
            E = np.asarray(E)
            if E.ndim == 1:
                if E.shape[0] != N:
                    raise ValueError("diagonal POVM vector has wrong length")
                M = np.diag(E.astype(float))
            else:
                if E.shape != (N, N):
                    raise ValueError("POVM matrix must be NxN")
                M = E.astype(complex)
            mats.append(M)
        # probabilities p_k = <psi|E_k|psi>
        probs = np.array([np.real(np.vdot(psi, M @ psi)) for M in mats], dtype=float)
        # normalize tiny negatives to zero
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if not np.isclose(s, 1.0, rtol=1e-6, atol=1e-6):
            # мягкая нормировка, если численно не идеальная сумма
            if s > 0:
                probs /= s
        # post-measure states: |psi_k'> = sqrt(E_k)|psi> / sqrt(p_k)
        post_regs = []
        for M, p in zip(mats, probs):
            if p <= 0:
                # возврат нулевого вектора (избежать деления)
                post_regs.append(PhaseRegister(np.zeros_like(psi)))
                continue
            # sqrt via eigendecomposition (assume Hermitian)
            try:
                w, V = np.linalg.eigh((M + M.conj().T) / 2.0)
                w = np.clip(np.real(w), 0.0, None)
                sqrtM = (V * np.sqrt(w)) @ V.conj().T
            except np.linalg.LinAlgError:
                # fallback: diagonal only
                if np.allclose(M, np.diag(np.diag(M))):
                    sqrtM = np.diag(np.sqrt(np.clip(np.real(np.diag(M)), 0.0, None)))
                else:
                    raise
            v = sqrtM @ psi
            if np.linalg.norm(v) == 0:
                post_regs.append(PhaseRegister(np.zeros_like(psi)))
            else:
                post = v / np.sqrt(p)
                post_regs.append(PhaseRegister(post))
        return probs, tuple(post_regs)


def deutsch_jozsa(oracle: Callable[[int], int], num_qubits: int) -> Tuple[str, float]:
    """
    Выполняет концептуальный алгоритм Дойча–Йожи на массиве фазовых состояний.

    Returns:
        (тип, p0): тип ∈ {"constant", "balanced"}, p0 — вероятность состояния y=0^n
    """
    reg = PhaseRegister.uniform(num_qubits)  # H^{⊗n} |0^n>
    # Оракул: флип фазы там, где f(x)=1
    reg.apply_oracle(oracle)
    # Обратное преобразование H^{⊗n}
    reg.hadamard_all()
    p0 = reg.measure_probability(0)
    guess = "constant" if np.isclose(p0, 1.0, rtol=1e-9, atol=1e-9) else "balanced"
    return guess, p0


