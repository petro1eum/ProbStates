# probstates/phase.py
"""
Реализация фазовых состояний (Уровень 4).
"""

from probstates.base import State
import numpy as np
import cmath
from typing import Union, Tuple, Optional


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
        
        Формула:
        (p₁, e^(iφ₁)) ⊕₄ (p₂, e^(iφ₂)) = (F(p₁,p₂,φ₁,φ₂), e^(iG(p₁,p₂,φ₁,φ₂)))
        
        где:
        F(p₁,p₂,φ₁,φ₂) = p₁ + p₂ + 2√(p₁p₂)cos(φ₁-φ₂)
        G(p₁,p₂,φ₁,φ₂) = arg(√p₁·e^(iφ₁) + √p₂·e^(iφ₂))
        
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
        
        # Вычисляем результирующую вероятность
        # F(p₁,p₂,φ₁,φ₂) = p₁ + p₂ + 2√(p₁p₂)cos(φ₁-φ₂)
        cos_term = np.cos(phi1 - phi2)
        p_result = p1 + p2 + 2 * np.sqrt(p1 * p2) * cos_term
        
        # Ограничиваем вероятность в пределах [0,1]
        p_result = max(0, min(1, p_result))
        
        # Вычисляем результирующую фазу
        # G(p₁,p₂,φ₁,φ₂) = arg(√p₁·e^(iφ₁) + √p₂·e^(iφ₂))
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
