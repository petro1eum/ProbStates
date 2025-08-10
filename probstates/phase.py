# probstates/phase.py
"""
Реализация фазовых состояний (Уровень 4).
"""

from probstates.base import State
import numpy as np
import cmath
from typing import Union, Tuple, Optional, Callable

# Глобальные настройки режима операции OR для фазовых состояний
# 'quant'  — квантово‑подобное сложение амплитуд (по умолчанию)
# 'opt'    — оптимизированное определение из статьи (с параметром Δφ)
# 'norm'   — альтернативное: F = min(1, F_quant), фаза как в 'quant'
# 'weight' — альтернативное: F = p1 ⊕₂ p2 + (2√(p1p2)cosΔφ)/(1+max(p1,p2)), фаза как в 'quant'
_PHASE_OR_MODE: str = 'quant'
_PHASE_OR_DELTA_PHI: float = np.pi / 2.0
_PHASE_OR_CUSTOM: Optional[Callable[[float, float, float, float], Tuple[float, float]]] = None

def set_phase_or_mode(mode: str = 'quant', delta_phi: float = np.pi/2) -> None:
    """
    Устанавливает режим операции OR (⊕₄) для фазовых состояний.

    Доступные режимы:
    - 'quant'  — квантово‑подобное сложение амплитуд:
      F_quant = p1 + p2 + 2√(p1p2)cos(φ1−φ2), фаза = arg(√p1e^{iφ1}+√p2e^{iφ2})
    - 'opt'    — оптимизированное определение из статьи:
      F_opt = p1 ⊕₂ p2 + 2(1−max(p1,p2))√(p1p2)cos(φ1−φ2), фаза = φ_avg + Δφ·sign(p1+p2−1)
    - 'norm'   — нормализованное квантово‑подобное:
      F_norm = min(1, F_quant), фаза как в 'quant'
    - 'weight' — взвешенный вариант:
      F_weight = p1 ⊕₂ p2 + (2√(p1p2)cos(φ1−φ2))/(1+max(p1,p2)), фаза как в 'quant'

    Args:
        mode: один из 'quant', 'opt', 'norm', 'weight'.
        delta_phi: параметр Δφ для режима 'opt'.
    """
    global _PHASE_OR_MODE, _PHASE_OR_DELTA_PHI
    if mode not in ('quant', 'opt', 'norm', 'weight', 'custom'):
        raise ValueError("phase OR mode must be 'quant', 'opt', 'norm', 'weight', or 'custom'")
    _PHASE_OR_MODE = mode
    _PHASE_OR_DELTA_PHI = float(delta_phi)

def set_phase_or_custom(custom_fn: Callable[[float, float, float, float], Tuple[float, float]]) -> None:
    """
    Устанавливает пользовательскую политику для операции OR (⊕₄).

    custom_fn: функция (p1, phi1, p2, phi2) -> (p_result, phi_result),
    должна возвращать p_result в [0,1] и фазу в радианах.
    После вызова этой функции активируйте режим через set_phase_or_mode('custom').
    """
    global _PHASE_OR_CUSTOM
    _PHASE_OR_CUSTOM = custom_fn

def get_phase_or_mode() -> str:
    """Возвращает текущий режим операции OR для фазовых состояний."""
    return _PHASE_OR_MODE


class PhaseState(State):
    """
    Фазовое состояние (Уровень 4).
    
    Представляет собой пару (p, e^(iφ)), где p ∈ [0,1] - вероятность,
    а φ ∈ [0, 2π) - фаза.
    """
    
    def __init__(self, probability: float, phase: float):
        """
        Инициализирует фазовое состояние.
        
        Args:
            probability: Вероятность (число в диапазоне [0,1]).
            phase: Фаза в радианах (число в диапазоне [0, 2π)).
            
        Raises:
            ValueError: Если вероятность не в диапазоне [0,1].
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in range [0,1], got {probability}")
        
        self._probability = float(probability)
        # Нормализуем фазу в диапазон [0, 2π)
        self._phase = float(phase) % (2 * np.pi)
    
    @property
    def probability(self) -> float:
        """Вероятностная составляющая фазового состояния."""
        return self._probability
    
    @property
    def phase(self) -> float:
        """Фаза состояния в радианах."""
        return self._phase
    
    @property
    def level(self) -> int:
        """Уровень в иерархии состояний (4 для фазовых состояний)."""
        return 4
    
    def __and__(self, other: 'PhaseState') -> 'PhaseState':
        """
        Операция AND (⊗₄) между двумя фазовыми состояниями.
        
        Формула: (p₁, e^(iφ₁)) ⊗₄ (p₂, e^(iφ₂)) = (p₁·p₂, e^(i(φ₁+φ₂)))
        
        Args:
            other: Другое фазовое состояние.
            
        Returns:
            Новое фазовое состояние - результат операции AND.
        """
        if not isinstance(other, PhaseState):
            return NotImplemented
        
        p_result = self.probability * other.probability
        phase_result = (self.phase + other.phase) % (2 * np.pi)
        
        return PhaseState(p_result, phase_result)
    
    def __or__(self, other: 'PhaseState') -> 'PhaseState':
        """
        Операция OR (⊕₄) между двумя фазовыми состояниями.
        
        По умолчанию используется квантово‑подобное сложение амплитуд (режим 'quant').
        Дополнительно поддерживается режим 'opt' из статьи (оптимизированное F/G).
        
        Args:
            other: Другое фазовое состояние.
            
        Returns:
            Новое фазовое состояние - результат операции OR.
        """
        if not isinstance(other, PhaseState):
            return NotImplemented
        
        # Извлекаем параметры
        p1, phi1 = self.probability, self.phase
        p2, phi2 = other.probability, other.phase

        mode = get_phase_or_mode()
        if mode == 'custom':
            if _PHASE_OR_CUSTOM is None:
                raise RuntimeError("Custom OR policy not set. Call set_phase_or_custom first.")
            p_result, phase_result = _PHASE_OR_CUSTOM(p1, phi1, p2, phi2)
            p_result = float(np.clip(p_result, 0.0, 1.0))
            phase_result = float(phase_result) % (2 * np.pi)
            return PhaseState(p_result, phase_result)
        if mode == 'opt':
            # F_opt = p1 ⊕2 p2 + K sqrt(p1 p2) cos(Δφ), K = 2(1 - max(p1,p2))
            # G_opt = phi1 (если p2=0) / phi2 (если p1=0) / phi_avg + Δφ * sign(p1+p2-1)
            p_or2 = p1 + p2 - p1 * p2
            K = 2.0 * (1.0 - max(p1, p2))
            cos_term = np.cos(phi1 - phi2)
            p_result = p_or2 + K * np.sqrt(p1 * p2) * cos_term
            # Безопасный клиппинг
            p_result = float(np.clip(p_result, 0.0, 1.0))

            if np.isclose(p2, 0.0):
                phase_result = phi1 % (2 * np.pi)
            elif np.isclose(p1, 0.0):
                phase_result = phi2 % (2 * np.pi)
            else:
                phi_avg = (phi1 + phi2) / 2.0
                s = 1.0 if (p1 + p2 - 1.0) >= 0.0 else -1.0
                phase_result = (phi_avg + s * _PHASE_OR_DELTA_PHI) % (2 * np.pi)

            return PhaseState(p_result, phase_result)
        elif mode == 'norm' or mode == 'weight' or mode == 'quant':
            # Режим 'quant' (по умолчанию): квантово‑подобное сложение амплитуд
            cos_term = np.cos(phi1 - phi2)
            p_quant = p1 + p2 + 2 * np.sqrt(p1 * p2) * cos_term
            if mode == 'norm':
                p_result = float(np.clip(min(1.0, p_quant), 0.0, 1.0))
            elif mode == 'weight':
                denom = 1.0 + max(p1, p2)
                p_or2 = p1 + p2 - p1 * p2
                p_result = p_or2 + (2.0 * np.sqrt(p1 * p2) * cos_term) / denom
                p_result = float(np.clip(p_result, 0.0, 1.0))
            else:
                p_result = float(np.clip(p_quant, 0.0, 1.0))

            z1 = cmath.rect(np.sqrt(p1), phi1)
            z2 = cmath.rect(np.sqrt(p2), phi2)
            z_sum = z1 + z2
            phase_result = cmath.phase(z_sum) % (2 * np.pi)

            return PhaseState(p_result, phase_result)
    
    def __invert__(self) -> 'PhaseState':
        """
        Операция NOT (¬₄) для фазового состояния.
        
        Формула: ¬₄(p, e^(iφ)) = (1-p, e^(i(φ+π)))
        
        Returns:
            Новое фазовое состояние - результат операции NOT.
        """
        phase_result = (self.phase + np.pi) % (2 * np.pi)
        return PhaseState(1 - self.probability, phase_result)

    # --- Измерения и вспомогательные функции уровня 4 ---

    def measure_sign(self) -> 'PBit':
        """
        Измерение в «знаковой» базе: возвращает P-бит с той же вероятностью p
        и полярностью sign(cos φ) согласно определению P_{4→3}.
        """
        # Локальный импорт, чтобы избежать циклических зависимостей
        from probstates.pbit import PBit
        sign = +1 if np.cos(self.phase) >= 0 else -1
        return PBit(self.probability, sign)

    def as_prob(self) -> 'ProbabilisticBit':
        """Возвращает вероятностный бит уровня 2 со значением p."""
        from probstates.probabilistic import ProbabilisticBit
        return ProbabilisticBit(self.probability)

    @staticmethod
    def phase_density(theta: float, p: float, phi: float) -> float:
        """
        Плотность фазового распределения f_p(θ, φ) = [1 + 2√(p(1-p)) cos(θ-φ)] / (2π).
        """
        return (1.0 + 2.0 * np.sqrt(p * (1.0 - p)) * np.cos(theta - phi)) / (2.0 * np.pi)
    
    def __eq__(self, other) -> bool:
        """
        Проверка равенства с другим объектом.
        
        Args:
            other: Объект для сравнения.
            
        Returns:
            True, если объекты равны, иначе False.
        """
        if isinstance(other, PhaseState):
            # Проверяем равенство вероятностей и фаз
            return (np.isclose(self.probability, other.probability) and 
                    (np.isclose(self.phase, other.phase) or 
                     np.isclose(self.phase % (2 * np.pi), other.phase % (2 * np.pi))))
        return False
    
    def __str__(self) -> str:
        """
        Строковое представление фазового состояния.
        
        Returns:
            Строка вида "(p, e^(iφ))".
        """
        return f"({self.probability:.4f}, e^(i{self.phase:.4f}))"
    
    def __repr__(self) -> str:
        """
        Полное строковое представление фазового состояния.
        
        Returns:
            Строка с описанием объекта.
        """
        return f"PhaseState({self.probability:.4f}, {self.phase:.4f})"
    
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        
        Returns:
            Словарь с полями 'level', 'probability' и 'phase'.
        """
        return {
            'level': self.level,
            'probability': self.probability,
            'phase': self.phase
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PhaseState':
        """
        Создает фазовое состояние из словаря.
        
        Args:
            data: Словарь с полями 'probability' и 'phase'.
            
        Returns:
            Новый объект PhaseState.
        """
        return cls(data['probability'], data['phase'])
    
    def to_complex(self) -> complex:
        """
        Преобразует фазовое состояние в комплексное число.
        
        Returns:
            Комплексное число вида √p · e^(iφ).
        """
        return cmath.rect(np.sqrt(self.probability), self.phase)
    
    @classmethod
    def from_complex(cls, z: complex) -> 'PhaseState':
        """
        Создает фазовое состояние из комплексного числа.
        
        Args:
            z: Комплексное число.
            
        Returns:
            Новый объект PhaseState с вероятностью |z|² и фазой arg(z).
        """
        probability = abs(z) ** 2
        # Ограничиваем вероятность в пределах [0,1]
        probability = max(0, min(1, probability))
        phase = cmath.phase(z) % (2 * np.pi)
        return cls(probability, phase)
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Tuple[int, float]:
        """
        Генерирует случайное значение согласно вероятности и фазе.
        
        Args:
            random_state: Объект генератора случайных чисел.
            
        Returns:
            Кортеж (value, phase), где value - 0 или 1, phase - фаза.
        """
        rng = random_state or np.random
        value = 1 if rng.random() < self.probability else 0
        return (value, self.phase)
