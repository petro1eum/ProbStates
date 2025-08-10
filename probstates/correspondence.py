"""
Numerical correspondence checker across levels: evaluates defect of
L_{n→m}(S ⊙_n T) vs L_{n→m}(S) ⊙_m L_{n→m}(T).
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


def correspondence_error(
    op_lower: Callable[[object, object], object],
    op_upper: Callable[[object, object], object],
    lift: Callable[[object], object],
    project_repr: Callable[[object], Tuple[float, float]],
    s_lower: object,
    t_lower: object,
) -> float:
    """
    Возвращает численную меру несоответствия между подъёмом результата операции
    и операцией над поднятыми состояниями. Для сравнения используем функцию
    project_repr, которая превращает состояние верхнего уровня в вектор признаков
    (например, (p, φ) или (p, sign)). Ошибка — евклидова норма разности векторов.
    """
    # L(S ⊙_n T)
    left = lift(op_lower(s_lower, t_lower))
    # L(S) ⊙_m L(T)
    right = op_upper(lift(s_lower), lift(t_lower))
    a = np.array(project_repr(left), dtype=float)
    b = np.array(project_repr(right), dtype=float)
    return float(np.linalg.norm(a - b))


