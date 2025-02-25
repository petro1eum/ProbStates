# probstates/classical.py
"""
Реализация классических битов (Уровень 1).
"""

from probstates.base import State
from typing import Union


class ClassicalBit(State):
    """
    Классический бит (Уровень 1).
    
    Представляет собой дискретное состояние 0 или 1.
    """
    
    def __init__(self, value: Union[bool, int]):
        """
        Инициализирует классический бит.
        
        Args:
            value: Значение бита (0 или 1, False или True).
        
        Raises:
            ValueError: Если значение не является 0, 1, False или True.
        """
        if isinstance(value, bool):
            self._value = int(value)
        elif value in [0, 1]:
            self._value = value
        else:
            raise ValueError(f"ClassicalBit value must be 0, 1, False or True, got {value}")
    
    @property
    def value(self) -> int:
        """Значение бита (0 или 1)."""
        return self._value
    
    @property
    def level(self) -> int:
        """Уровень в иерархии состояний (1 для классических битов)."""
        return 1
    
    def __and__(self, other: 'ClassicalBit') -> 'ClassicalBit':
        """
        Логическое AND (∧) между двумя классическими битами.
        
        Args:
            other: Другой классический бит.
            
        Returns:
            Новый классический бит - результат операции AND.
        """
        if not isinstance(other, ClassicalBit):
            return NotImplemented
        return ClassicalBit(self.value and other.value)
    
    def __or__(self, other: 'ClassicalBit') -> 'ClassicalBit':
        """
        Логическое OR (∨) между двумя классическими битами.
        
        Args:
            other: Другой классический бит.
            
        Returns:
            Новый классический бит - результат операции OR.
        """
        if not isinstance(other, ClassicalBit):
            return NotImplemented
        return ClassicalBit(self.value or other.value)
    
    def __invert__(self) -> 'ClassicalBit':
        """
        Логическое NOT (¬) классического бита.
        
        Returns:
            Новый классический бит - результат операции NOT.
        """
        return ClassicalBit(1 - self.value)
    
    def __eq__(self, other) -> bool:
        """
        Проверка равенства с другим объектом.
        
        Args:
            other: Объект для сравнения.
            
        Returns:
            True, если объекты равны, иначе False.
        """
        if isinstance(other, ClassicalBit):
            return self.value == other.value
        elif isinstance(other, (int, bool)):
            return self.value == (1 if other else 0)
        return False
    
    def __str__(self) -> str:
        """
        Строковое представление классического бита.
        
        Returns:
            Строка '0' или '1'.
        """
        return str(self.value)
    
    def __repr__(self) -> str:
        """
        Полное строковое представление классического бита.
        
        Returns:
            Строка с описанием объекта.
        """
        return f"ClassicalBit({self.value})"
    
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        
        Returns:
            Словарь с полями 'level' и 'value'.
        """
        return {'level': self.level, 'value': self.value}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ClassicalBit':
        """
        Создает классический бит из словаря.
        
        Args:
            data: Словарь с полем 'value'.
            
        Returns:
            Новый объект ClassicalBit.
        """
        return cls(data['value'])
