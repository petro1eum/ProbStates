# probstates/base.py
"""
Базовые абстрактные классы для формализма вероятностных состояний.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, TypeVar, Generic, Union, Tuple

# Тип для обобщения состояний
S = TypeVar('S', bound='State')

class State(ABC):
    """
    Абстрактный базовый класс для всех типов состояний.
    Определяет общий интерфейс для всех уровней иерархии.
    """
    
    @property
    @abstractmethod
    def level(self) -> int:
        """
        Возвращает уровень состояния в иерархии (1-5).
        """
        pass
    
    @abstractmethod
    def __and__(self, other: S) -> S:
        """
        Реализует операцию AND (⊗) для состояний.
        Аналог логического AND для классических битов.
        """
        pass
    
    @abstractmethod
    def __or__(self, other: S) -> S:
        """
        Реализует операцию OR (⊕) для состояний.
        Аналог логического OR для классических битов.
        """
        pass
    
    @abstractmethod
    def __invert__(self) -> S:
        """
        Реализует операцию NOT (¬) для состояний.
        Аналог логического NOT для классических битов.
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """
        Строковое представление состояния.
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Полное строковое представление состояния.
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        Используется для сериализации и визуализации.
        """
        pass


class Operator(ABC, Generic[S]):
    """
    Абстрактный базовый класс для операторов, действующих на состояния.
    """
    
    @abstractmethod
    def apply(self, state: S) -> Any:
        """
        Применяет оператор к состоянию.
        """
        pass
    
    def __call__(self, state: S) -> Any:
        """
        Синтаксический сахар для применения оператора.
        """
        return self.apply(state)


class LiftingOperator(Operator):
    """
    Оператор подъема состояния на следующий уровень иерархии.
    """
    
    def __init__(self, from_level: int, to_level: int):
        if from_level >= to_level:
            raise ValueError(f"Lifting operator requires from_level ({from_level}) < to_level ({to_level})")
        self.from_level = from_level
        self.to_level = to_level
    
    @abstractmethod
    def apply(self, state: S) -> Any:
        """
        Поднимает состояние на более высокий уровень.
        """
        pass


class ProjectionOperator(Operator):
    """
    Оператор проекции состояния на нижний уровень иерархии.
    """
    
    def __init__(self, from_level: int, to_level: int):
        if from_level <= to_level:
            raise ValueError(f"Projection operator requires from_level ({from_level}) > to_level ({to_level})")
        self.from_level = from_level
        self.to_level = to_level
    
    @abstractmethod
    def apply(self, state: S) -> Any:
        """
        Проецирует состояние на более низкий уровень.
        """
        pass
