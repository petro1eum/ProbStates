# probstates/probabilistic.py
"""
Реализация вероятностных битов (Уровень 2).
"""

from probstates.base import State
import numpy as np
from typing import Union, Optional


class ProbabilisticBit(State):
    """
    Вероятностный бит (Уровень 2).
    
    Представляет собой вероятность p ∈ [0,1], где p - вероятность
    того, что бит имеет значение 1.
    """
    
    def __init__(self, probability: float):
        """
        Инициализирует вероятностный бит.
        
        Args:
            probability: Вероятность значения 1 (число в диапазоне [0,1]).
            
        Raises:
            ValueError: Если вероятность не в диапазоне [0,1].
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in range [0,1], got {probability}")
        self._probability = float(probability)
    
    @property
    def probability(self) -> float:
        """Вероятность того, что бит имеет значение 1."""
        return self._probability
    
    @property
    def level(self) -> int:
        """Уровень в иерархии состояний (2 для вероятностных битов)."""
        return 2
    
    def __and__(self, other: 'ProbabilisticBit') -> 'ProbabilisticBit':
        """
        Вероятностное AND (⊗₂) между двумя вероятностными битами.
        
        Формула: p₁ ⊗₂ p₂ = p₁ · p₂
        
        Args:
            other: Другой вероятностный бит.
            
        Returns:
            Новый вероятностный бит - результат операции AND.
        """
        if not isinstance(other, ProbabilisticBit):
            return NotImplemented
        return ProbabilisticBit(self.probability * other.probability)
    
    def __or__(self, other: 'ProbabilisticBit') -> 'ProbabilisticBit':
        """
        Вероятностное OR (⊕₂) между двумя вероятностными битами.
        
        Формула: p₁ ⊕₂ p₂ = p₁ + p₂ - p₁·p₂
        
        Args:
            other: Другой вероятностный бит.
            
        Returns:
            Новый вероятностный бит - результат операции OR.
        """
        if not isinstance(other, ProbabilisticBit):
            return NotImplemented
        p1, p2 = self.probability, other.probability
        return ProbabilisticBit(p1 + p2 - p1 * p2)
    
    def __invert__(self) -> 'ProbabilisticBit':
        """
        Вероятностное NOT (¬₂) вероятностного бита.
        
        Формула: ¬₂p = 1 - p
        
        Returns:
            Новый вероятностный бит - результат операции NOT.
        """
        return ProbabilisticBit(1 - self.probability)
    
    def __eq__(self, other) -> bool:
        """
        Проверка равенства с другим объектом.
        
        Args:
            other: Объект для сравнения.
            
        Returns:
            True, если объекты равны с точностью до погрешности, иначе False.
        """
        if isinstance(other, ProbabilisticBit):
            return np.isclose(self.probability, other.probability)
        elif isinstance(other, (int, float)):
            return np.isclose(self.probability, float(other))
        return False
    
    def __str__(self) -> str:
        """
        Строковое представление вероятностного бита.
        
        Returns:
            Строка с вероятностью.
        """
        return f"{self.probability:.4f}"
    
    def __repr__(self) -> str:
        """
        Полное строковое представление вероятностного бита.
        
        Returns:
            Строка с описанием объекта.
        """
        return f"ProbabilisticBit({self.probability:.4f})"
    
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        
        Returns:
            Словарь с полями 'level' и 'probability'.
        """
        return {'level': self.level, 'probability': self.probability}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProbabilisticBit':
        """
        Создает вероятностный бит из словаря.
        
        Args:
            data: Словарь с полем 'probability'.
            
        Returns:
            Новый объект ProbabilisticBit.
        """
        return cls(data['probability'])
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> int:
        """
        Генерирует случайное значение согласно вероятности.
        
        Args:
            random_state: Объект генератора случайных чисел.
            
        Returns:
            0 или 1 с вероятностью, соответствующей состоянию.
        """
        rng = random_state or np.random
        return 1 if rng.random() < self.probability else 0
