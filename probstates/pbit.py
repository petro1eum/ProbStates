# probstates/pbit.py
"""
Реализация P-битов (Уровень 3).
"""

from probstates.base import State
import numpy as np
from typing import Union, Tuple, Optional


class PBit(State):
    """
    P-бит (Уровень 3).
    
    Представляет собой пару (p, s), где p ∈ [0,1] - вероятность,
    а s ∈ {+1, -1} - полярность (знак).
    """
    
    def __init__(self, probability: float, sign: int):
        """
        Инициализирует P-бит.
        
        Args:
            probability: Вероятность (число в диапазоне [0,1]).
            sign: Полярность (+1 или -1).
            
        Raises:
            ValueError: Если вероятность не в диапазоне [0,1] или знак не +1/-1.
        """
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability must be in range [0,1], got {probability}")
        if sign not in [-1, 1]:
            raise ValueError(f"Sign must be either -1 or 1, got {sign}")
        
        self._probability = float(probability)
        self._sign = int(sign)
    
    @property
    def probability(self) -> float:
        """Вероятностная составляющая P-бита."""
        return self._probability
    
    @property
    def sign(self) -> int:
        """Полярность P-бита (+1 или -1)."""
        return self._sign
    
    @property
    def level(self) -> int:
        """Уровень в иерархии состояний (3 для P-битов)."""
        return 3
    
    def __and__(self, other: 'PBit') -> 'PBit':
        """
        Операция AND (⊗₃) между двумя P-битами.
        
        Формула: (p₁, s₁) ⊗₃ (p₂, s₂) = (p₁·p₂, s₁·s₂)
        
        Args:
            other: Другой P-бит.
            
        Returns:
            Новый P-бит - результат операции AND.
        """
        if not isinstance(other, PBit):
            return NotImplemented
        
        p_result = self.probability * other.probability
        s_result = self.sign * other.sign
        
        return PBit(p_result, s_result)
    
    def __or__(self, other: 'PBit') -> 'PBit':
        """
        Операция OR (⊕₃) между двумя P-битами.
        
        Формула: (p₁, s₁) ⊕₃ (p₂, s₂) = (p₁ + p₂ - p₁·p₂, s₁·s₂·sign(p₁ + p₂ - 1))
        
        Args:
            other: Другой P-бит.
            
        Returns:
            Новый P-бит - результат операции OR.
        """
        if not isinstance(other, PBit):
            return NotImplemented
        
        p1, s1 = self.probability, self.sign
        p2, s2 = other.probability, other.sign
        
        p_result = p1 + p2 - p1 * p2
        phase_term = p1 + p2 - 1
        s_result = s1 * s2 * (1 if phase_term >= 0 else -1)
        
        return PBit(p_result, s_result)
    
    def __invert__(self) -> 'PBit':
        """
        Операция NOT (¬₃) для P-бита.
        
        Формула: ¬₃(p, s) = (1-p, -s)
        
        Returns:
            Новый P-бит - результат операции NOT.
        """
        return PBit(1 - self.probability, -self.sign)
    
    def __eq__(self, other) -> bool:
        """
        Проверка равенства с другим объектом.
        
        Args:
            other: Объект для сравнения.
            
        Returns:
            True, если объекты равны, иначе False.
        """
        if isinstance(other, PBit):
            return (np.isclose(self.probability, other.probability) and 
                    self.sign == other.sign)
        return False
    
    def __str__(self) -> str:
        """
        Строковое представление P-бита.
        
        Returns:
            Строка вида "(p, s)".
        """
        return f"({self.probability:.4f}, {self.sign:+d})"
    
    def __repr__(self) -> str:
        """
        Полное строковое представление P-бита.
        
        Returns:
            Строка с описанием объекта.
        """
        return f"PBit({self.probability:.4f}, {self.sign})"
    
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        
        Returns:
            Словарь с полями 'level', 'probability' и 'sign'.
        """
        return {
            'level': self.level,
            'probability': self.probability,
            'sign': self.sign
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PBit':
        """
        Создает P-бит из словаря.
        
        Args:
            data: Словарь с полями 'probability' и 'sign'.
            
        Returns:
            Новый объект PBit.
        """
        return cls(data['probability'], data['sign'])
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        """
        Генерирует случайное значение согласно вероятности и полярности.
        
        Args:
            random_state: Объект генератора случайных чисел.
            
        Returns:
            Кортеж (value, sign), где value - 0 или 1, sign - полярность.
        """
        rng = random_state or np.random
        value = 1 if rng.random() < self.probability else 0
        return (value, self.sign)
