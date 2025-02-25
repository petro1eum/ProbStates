# probstates/operators.py
"""
Операторы перехода между различными уровнями иерархии вероятностных состояний.
"""

from probstates.base import LiftingOperator, ProjectionOperator, State
from probstates.classical import ClassicalBit
from probstates.probabilistic import ProbabilisticBit
from probstates.pbit import PBit
from probstates.phase import PhaseState
from probstates.quantum import QuantumState

import numpy as np
from typing import Union, Type, TypeVar, Any

# Типы для аннотаций
S = TypeVar('S', bound=State)


class Lift1to2(LiftingOperator):
    """
    Оператор подъема с уровня 1 (классические биты) на уровень 2 (вероятностные биты).
    """
    
    def __init__(self):
        super().__init__(1, 2)
    
    def apply(self, state: ClassicalBit) -> ProbabilisticBit:
        """
        Поднимает классический бит до вероятностного бита.
        
        Формула: L₁→₂(b) = b
        
        Args:
            state: Классический бит.
            
        Returns:
            Вероятностный бит с вероятностью 0 или 1.
        """
        if not isinstance(state, ClassicalBit):
            raise TypeError(f"Expected ClassicalBit, got {type(state).__name__}")
        
        return ProbabilisticBit(float(state.value))


class Lift2to3(LiftingOperator):
    """
    Оператор подъема с уровня 2 (вероятностные биты) на уровень 3 (P-биты).
    """
    
    def __init__(self):
        super().__init__(2, 3)
    
    def apply(self, state: ProbabilisticBit) -> PBit:
        """
        Поднимает вероятностный бит до P-бита.
        
        Формула: L₂→₃(p) = (p, +1)
        
        Args:
            state: Вероятностный бит.
            
        Returns:
            P-бит с той же вероятностью и положительной полярностью.
        """
        if not isinstance(state, ProbabilisticBit):
            raise TypeError(f"Expected ProbabilisticBit, got {type(state).__name__}")
        
        return PBit(state.probability, +1)


class Lift3to4(LiftingOperator):
    """
    Оператор подъема с уровня 3 (P-биты) на уровень 4 (фазовые состояния).
    """
    
    def __init__(self):
        super().__init__(3, 4)
    
    def apply(self, state: PBit) -> PhaseState:
        """
        Поднимает P-бит до фазового состояния.
        
        Формула: L₃→₄(p, s) = (p, e^(iφₛ)), где φₛ = 0 при s = +1, φₛ = π при s = -1.
        
        Args:
            state: P-бит.
            
        Returns:
            Фазовое состояние с той же вероятностью и фазой 0 или π.
        """
        if not isinstance(state, PBit):
            raise TypeError(f"Expected PBit, got {type(state).__name__}")
        
        # Преобразуем полярность в фазу
        phase = 0.0 if state.sign == 1 else np.pi
        
        return PhaseState(state.probability, phase)


class Lift4to5(LiftingOperator):
    """
    Оператор подъема с уровня 4 (фазовые состояния) на уровень 5 (квантовые состояния).
    """
    
    def __init__(self):
        super().__init__(4, 5)
    
    def apply(self, state: PhaseState) -> QuantumState:
        """
        Поднимает фазовое состояние до квантового состояния.
        
        Формула: L₄→₅(p, e^(iφ)) = √p·e^(iφ)|0⟩ + √(1-p)|1⟩
        
        Args:
            state: Фазовое состояние.
            
        Returns:
            Квантовое состояние с соответствующими амплитудами.
        """
        if not isinstance(state, PhaseState):
            raise TypeError(f"Expected PhaseState, got {type(state).__name__}")
        
        # Извлекаем параметры
        p, phi = state.probability, state.phase
        
        # Создаем амплитуды для квантового состояния
        alpha = np.sqrt(p) * np.exp(1j * phi)
        beta = np.sqrt(1 - p)
        
        return QuantumState([alpha, beta])


class Project2to1(ProjectionOperator):
    """
    Оператор проекции с уровня 2 (вероятностные биты) на уровень 1 (классические биты).
    """
    
    def __init__(self):
        super().__init__(2, 1)
    
    def apply(self, state: ProbabilisticBit) -> ClassicalBit:
        """
        Проецирует вероятностный бит на классический бит.
        
        Формула: P₂→₁(p) = 1 при p ≥ 0.5, P₂→₁(p) = 0 при p < 0.5
        
        Args:
            state: Вероятностный бит.
            
        Returns:
            Классический бит, округленный по вероятности.
        """
        if not isinstance(state, ProbabilisticBit):
            raise TypeError(f"Expected ProbabilisticBit, got {type(state).__name__}")
        
        return ClassicalBit(1 if state.probability >= 0.5 else 0)


class Project3to2(ProjectionOperator):
    """
    Оператор проекции с уровня 3 (P-биты) на уровень 2 (вероятностные биты).
    """
    
    def __init__(self):
        super().__init__(3, 2)
    
    def apply(self, state: PBit) -> ProbabilisticBit:
        """
        Проецирует P-бит на вероятностный бит.
        
        Формула: P₃→₂(p, s) = p при s = +1, P₃→₂(p, s) = 1-p при s = -1
        
        Args:
            state: P-бит.
            
        Returns:
            Вероятностный бит с учетом полярности.
        """
        if not isinstance(state, PBit):
            raise TypeError(f"Expected PBit, got {type(state).__name__}")
        
        # Для отрицательной полярности инвертируем вероятность
        p = state.probability if state.sign == 1 else 1 - state.probability
        
        return ProbabilisticBit(p)


class Project4to3(ProjectionOperator):
    """
    Оператор проекции с уровня 4 (фазовые состояния) на уровень 3 (P-биты).
    """
    
    def __init__(self):
        super().__init__(4, 3)
    
    def apply(self, state: PhaseState) -> PBit:
        """
        Проецирует фазовое состояние на P-бит.
        
        Формула: P₄→₃(p, e^(iφ)) = (p, sign(cos(φ)))
        
        Args:
            state: Фазовое состояние.
            
        Returns:
            P-бит с той же вероятностью и знаком, зависящим от фазы.
        """
        if not isinstance(state, PhaseState):
            raise TypeError(f"Expected PhaseState, got {type(state).__name__}")
        
        # Определяем знак исходя из косинуса фазы
        sign = 1 if np.cos(state.phase) >= 0 else -1
        
        return PBit(state.probability, sign)


class Project5to4(ProjectionOperator):
    """
    Оператор проекции с уровня 5 (квантовые состояния) на уровень 4 (фазовые состояния).
    """
    
    def __init__(self):
        super().__init__(5, 4)
    
    def apply(self, state: QuantumState) -> PhaseState:
        """
        Проецирует квантовое состояние на фазовое состояние.
        
        Формула: P₅→₄(|ψ⟩) = (|⟨0|ψ⟩|², arg(⟨0|ψ⟩))
        
        Args:
            state: Квантовое состояние.
            
        Returns:
            Фазовое состояние, соответствующее амплитуде |0⟩.
        """
        if not isinstance(state, QuantumState):
            raise TypeError(f"Expected QuantumState, got {type(state).__name__}")
        
        # Извлекаем амплитуду |0⟩
        alpha = state.amplitudes[0]
        
        # Вычисляем вероятность и фазу
        probability = abs(alpha) ** 2
        phase = np.angle(alpha) % (2 * np.pi)
        
        return PhaseState(probability, phase)


# Словарь операторов подъема
_LIFT_OPERATORS = {
    (1, 2): Lift1to2(),
    (2, 3): Lift2to3(),
    (3, 4): Lift3to4(),
    (4, 5): Lift4to5(),
    # Композиции
    (1, 3): None,  # Будет создана динамически
    (1, 4): None,  # Будет создана динамически
    (1, 5): None,  # Будет создана динамически
    (2, 4): None,  # Будет создана динамически
    (2, 5): None,  # Будет создана динамически
    (3, 5): None   # Будет создана динамически
}

# Словарь операторов проекции
_PROJECT_OPERATORS = {
    (2, 1): Project2to1(),
    (3, 2): Project3to2(),
    (4, 3): Project4to3(),
    (5, 4): Project5to4(),
    # Композиции
    (3, 1): None,  # Будет создана динамически
    (4, 1): None,  # Будет создана динамически
    (4, 2): None,  # Будет создана динамически
    (5, 1): None,  # Будет создана динамически
    (5, 2): None,  # Будет создана динамически
    (5, 3): None   # Будет создана динамически
}

# Инициализация композиций операторов
def _init_compositions():
    """Инициализирует композиции операторов."""
    # Композиции операторов подъема
    _LIFT_OPERATORS[(1, 3)] = lambda s: _LIFT_OPERATORS[(2, 3)].apply(_LIFT_OPERATORS[(1, 2)].apply(s))
    _LIFT_OPERATORS[(1, 4)] = lambda s: _LIFT_OPERATORS[(3, 4)].apply(_LIFT_OPERATORS[(1, 3)](s))
    _LIFT_OPERATORS[(1, 5)] = lambda s: _LIFT_OPERATORS[(4, 5)].apply(_LIFT_OPERATORS[(1, 4)](s))
    _LIFT_OPERATORS[(2, 4)] = lambda s: _LIFT_OPERATORS[(3, 4)].apply(_LIFT_OPERATORS[(2, 3)].apply(s))
    _LIFT_OPERATORS[(2, 5)] = lambda s: _LIFT_OPERATORS[(4, 5)].apply(_LIFT_OPERATORS[(2, 4)](s))
    _LIFT_OPERATORS[(3, 5)] = lambda s: _LIFT_OPERATORS[(4, 5)].apply(_LIFT_OPERATORS[(3, 4)].apply(s))
    
    # Композиции операторов проекции
    _PROJECT_OPERATORS[(3, 1)] = lambda s: _PROJECT_OPERATORS[(2, 1)].apply(_PROJECT_OPERATORS[(3, 2)].apply(s))
    _PROJECT_OPERATORS[(4, 1)] = lambda s: _PROJECT_OPERATORS[(2, 1)].apply(_PROJECT_OPERATORS[(4, 2)](s))
    _PROJECT_OPERATORS[(4, 2)] = lambda s: _PROJECT_OPERATORS[(3, 2)].apply(_PROJECT_OPERATORS[(4, 3)].apply(s))
    _PROJECT_OPERATORS[(5, 1)] = lambda s: _PROJECT_OPERATORS[(2, 1)].apply(_PROJECT_OPERATORS[(5, 2)](s))
    _PROJECT_OPERATORS[(5, 2)] = lambda s: _PROJECT_OPERATORS[(3, 2)].apply(_PROJECT_OPERATORS[(5, 3)](s))
    _PROJECT_OPERATORS[(5, 3)] = lambda s: _PROJECT_OPERATORS[(4, 3)].apply(_PROJECT_OPERATORS[(5, 4)].apply(s))

# Инициализируем композиции при импорте модуля
_init_compositions()

def lift(state: State, to_level: int) -> State:
    """
    Поднимает состояние до указанного уровня.
    
    Args:
        state: Исходное состояние.
        to_level: Целевой уровень (должен быть выше исходного).
        
    Returns:
        Поднятое состояние указанного уровня.
        
    Raises:
        ValueError: Если целевой уровень не выше исходного или не в диапазоне [1,5].
    """
    from_level = state.level
    
    if not 1 <= to_level <= 5:
        raise ValueError(f"Target level must be in range [1,5], got {to_level}")
    
    if from_level >= to_level:
        raise ValueError(f"Target level ({to_level}) must be higher than source level ({from_level})")
    
    op = _LIFT_OPERATORS.get((from_level, to_level))
    if op is None:
        raise ValueError(f"No lifting operator defined for levels {from_level} -> {to_level}")
    
    return op(state)

def project(state: State, to_level: int) -> State:
    """
    Проецирует состояние на указанный уровень.
    
    Args:
        state: Исходное состояние.
        to_level: Целевой уровень (должен быть ниже исходного).
        
    Returns:
        Проекция состояния на указанный уровень.
        
    Raises:
        ValueError: Если целевой уровень не ниже исходного или не в диапазоне [1,5].
    """
    from_level = state.level
    
    if not 1 <= to_level <= 5:
        raise ValueError(f"Target level must be in range [1,5], got {to_level}")
    
    if from_level <= to_level:
        raise ValueError(f"Target level ({to_level}) must be lower than source level ({from_level})")
    
    op = _PROJECT_OPERATORS.get((from_level, to_level))
    if op is None:
        raise ValueError(f"No projection operator defined for levels {from_level} -> {to_level}")
    
    return op(state)
