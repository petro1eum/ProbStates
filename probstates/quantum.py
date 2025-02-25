# probstates/quantum.py
"""
Реализация квантовых состояний (Уровень 5).
"""

from probstates.base import State
import numpy as np
import cmath
from typing import Union, List, Tuple, Optional


class QuantumState(State):
    """
    Квантовое состояние (Уровень 5).
    
    Представляет собой вектор состояния |ψ⟩ в комплексном пространстве.
    В данной реализации ограничиваемся одним кубитом для простоты,
    т.е. |ψ⟩ = α|0⟩ + β|1⟩, где |α|² + |β|² = 1.
    """
    
    def __init__(self, amplitudes: Union[List[complex], np.ndarray]):
        """
        Инициализирует квантовое состояние.
        
        Args:
            amplitudes: Список или массив комплексных амплитуд [α, β].
            
        Raises:
            ValueError: Если длина массива не равна 2 или нарушена нормировка.
        """
        if len(amplitudes) != 2:
            raise ValueError(f"For qubit state, amplitudes must have length 2, got {len(amplitudes)}")
        
        # Преобразуем в numpy массив
        self._amplitudes = np.array(amplitudes, dtype=complex)
        
        # Проверяем нормировку
        norm = np.linalg.norm(self._amplitudes)
        if not np.isclose(norm, 1.0, rtol=1e-5):
            # Нормализуем состояние
            self._amplitudes = self._amplitudes / norm
    
    @property
    def amplitudes(self) -> np.ndarray:
        """Амплитуды квантового состояния."""
        return self._amplitudes.copy()
    
    @property
    def level(self) -> int:
        """Уровень в иерархии состояний (5 для квантовых состояний)."""
        return 5
    
    def __and__(self, other: 'QuantumState') -> 'QuantumState':
        """
        Операция AND (⊗₅) между двумя квантовыми состояниями.
        
        В квантовой механике это соответствует тензорному произведению состояний.
        Однако, для упрощения и сохранения одинаковой размерности, 
        мы моделируем эффект проекции на состояние |11⟩.
        
        Args:
            other: Другое квантовое состояние.
            
        Returns:
            Новое квантовое состояние - результат операции.
        """
        if not isinstance(other, QuantumState):
            return NotImplemented
        
        # Для простоты моделируем эффект проецирования на подпространство |11⟩
        # Вероятность измерить |1⟩ для self
        p1 = abs(self._amplitudes[1]) ** 2
        # Вероятность измерить |1⟩ для other
        p2 = abs(other._amplitudes[1]) ** 2
        
        # Вероятность обоих состояний в |1⟩
        p_and = p1 * p2
        
        # Создаем новое состояние, приближенно представляющее результат AND
        if np.isclose(p_and, 0):
            # Если вероятность нулевая, возвращаем |0⟩
            return QuantumState([1, 0])
        elif np.isclose(p_and, 1):
            # Если вероятность единичная, возвращаем |1⟩
            return QuantumState([0, 1])
        else:
            # Иначе создаем суперпозицию
            alpha = np.sqrt(1 - p_and)
            beta = np.sqrt(p_and)
            return QuantumState([alpha, beta])
    
    def __or__(self, other: 'QuantumState') -> 'QuantumState':
        """
        Операция OR (⊕₅) между двумя квантовыми состояниями.
        
        В квантовой механике это соответствует суперпозиции состояний.
        Для упрощения, мы создаем равную суперпозицию двух состояний.
        
        Args:
            other: Другое квантовое состояние.
            
        Returns:
            Новое квантовое состояние - результат операции.
        """
        if not isinstance(other, QuantumState):
            return NotImplemented
        
        # Создаем равную суперпозицию состояний
        new_state = (self._amplitudes + other._amplitudes) / np.sqrt(2)
        
        # Нормализуем
        new_state = new_state / np.linalg.norm(new_state)
        
        return QuantumState(new_state)
    
    def __invert__(self) -> 'QuantumState':
        """
        Операция NOT (¬₅) для квантового состояния.
        
        В квантовой механике это соответствует инвертированию состояния.
        Для кубита это соответствует применению оператора X (Паули-X).
        
        Returns:
            Новое квантовое состояние - результат операции.
        """
        # Матрица Паули-X (оператор NOT)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Применяем оператор X к состоянию
        new_state = X @ self._amplitudes
        
        return QuantumState(new_state)
    
    def __eq__(self, other) -> bool:
        """
        Проверка равенства с другим объектом.
        
        Args:
            other: Объект для сравнения.
            
        Returns:
            True, если объекты равны, иначе False.
        """
        if isinstance(other, QuantumState):
            # Проверяем равенство амплитуд с точностью до глобальной фазы
            phase_diff = np.angle(self._amplitudes[0] / other._amplitudes[0]) if (
                abs(self._amplitudes[0]) > 1e-10 and abs(other._amplitudes[0]) > 1e-10) else 0
            
            corrected_other = other._amplitudes * np.exp(-1j * phase_diff)
            return np.allclose(self._amplitudes, corrected_other)
        
        return False
    
    def __str__(self) -> str:
        """
        Строковое представление квантового состояния.
        
        Returns:
            Строка вида "α|0⟩ + β|1⟩".
        """
        alpha, beta = self._amplitudes
        
        alpha_str = f"{alpha.real:.4f}"
        if abs(alpha.imag) > 1e-10:
            alpha_str += f"{'+' if alpha.imag > 0 else ''}{alpha.imag:.4f}i"
        
        beta_str = f"{beta.real:.4f}"
        if abs(beta.imag) > 1e-10:
            beta_str += f"{'+' if beta.imag > 0 else ''}{beta.imag:.4f}i"
        
        return f"{alpha_str}|0⟩ + {beta_str}|1⟩"
    
    def __repr__(self) -> str:
        """
        Полное строковое представление квантового состояния.
        
        Returns:
            Строка с описанием объекта.
        """
        return f"QuantumState([{self._amplitudes[0]}, {self._amplitudes[1]}])"
    
    def to_dict(self) -> dict:
        """
        Представление состояния в виде словаря.
        
        Returns:
            Словарь с полями 'level' и 'amplitudes'.
        """
        return {
            'level': self.level,
            'amplitudes': [
                {'real': self._amplitudes[0].real, 'imag': self._amplitudes[0].imag},
                {'real': self._amplitudes[1].real, 'imag': self._amplitudes[1].imag}
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'QuantumState':
        """
        Создает квантовое состояние из словаря.
        
        Args:
            data: Словарь с полем 'amplitudes'.
            
        Returns:
            Новый объект QuantumState.
        """
        amplitudes = [
            complex(a['real'], a['imag']) for a in data['amplitudes']
        ]
        return cls(amplitudes)
    
    def measure(self, random_state: Optional[np.random.RandomState] = None) -> int:
        """
        Измеряет квантовое состояние в стандартном базисе.
        
        Args:
            random_state: Объект генератора случайных чисел.
            
        Returns:
            0 или 1 согласно распределению вероятностей |α|² и |β|².
        """
        rng = random_state or np.random
        probabilities = np.abs(self._amplitudes) ** 2
        return rng.choice([0, 1], p=probabilities)
    
    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> 'QuantumState':
        """
        Создает квантовое состояние из координат на сфере Блоха.
        
        Args:
            theta: Полярный угол [0, π].
            phi: Азимутальный угол [0, 2π).
            
        Returns:
            Новый объект QuantumState.
        """
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return cls([alpha, beta])
    
    def to_bloch(self) -> Tuple[float, float]:
        """
        Преобразует квантовое состояние в координаты на сфере Блоха.
        
        Returns:
            Кортеж (theta, phi) - полярный и азимутальный углы.
        """
        alpha, beta = self._amplitudes
        
        # Удаляем глобальную фазу
        if abs(alpha) > 1e-10:
            phase = np.angle(alpha)
            alpha *= np.exp(-1j * phase)
            beta *= np.exp(-1j * phase)
        
        theta = 2 * np.arccos(abs(alpha))
        phi = np.angle(beta) if abs(beta) > 1e-10 else 0
        
        return (theta, phi)
