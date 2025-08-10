# probstates/visualization.py
"""
Функции для визуализации различных вероятностных состояний.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from typing import Union, List, Optional, Tuple, Any

from probstates.base import State
from probstates.classical import ClassicalBit
from probstates.probabilistic import ProbabilisticBit
from probstates.pbit import PBit
from probstates.phase import PhaseState
from probstates.quantum import QuantumState



def visualize_classical_bit(state: ClassicalBit, ax: Optional[plt.Axes] = None, 
                            title: str = "Классический бит") -> plt.Axes:
    """
    Визуализирует классический бит.
    
    Args:
        state: Объект ClassicalBit для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика.
        
    Returns:
        Объект осей matplotlib с визуализацией.
    """
    if not isinstance(state, ClassicalBit):
        raise TypeError(f"Expected ClassicalBit, got {type(state).__name__}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    
    # Рисуем линейку
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.plot([0, 1], [0, 0], 'k-', linewidth=2)
    
    # Отмечаем 0 и 1
    ax.plot(0, 0, 'o', markersize=10, color='lightgray')
    ax.plot(1, 0, 'o', markersize=10, color='lightgray')
    ax.text(0, -0.2, '0', fontsize=12, ha='center')
    ax.text(1, -0.2, '1', fontsize=12, ha='center')
    
    # Выделяем текущее значение
    color = 'navy' if state.value == 0 else 'darkgreen'
    ax.plot(state.value, 0, 'o', markersize=15, color=color)
    ax.text(state.value, 0.2, str(state.value), fontsize=14, 
            ha='center', weight='bold', color=color)
    
    # Оформление
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax


def visualize_probabilistic_bit(state: ProbabilisticBit, ax: Optional[plt.Axes] = None,
                               title: str = "Вероятностный бит") -> plt.Axes:
    """
    Визуализирует вероятностный бит.
    
    Args:
        state: Объект ProbabilisticBit для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика.
        
    Returns:
        Объект осей matplotlib с визуализацией.
    """
    if not isinstance(state, ProbabilisticBit):
        raise TypeError(f"Expected ProbabilisticBit, got {type(state).__name__}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    
    # Рисуем шкалу вероятности
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.plot([0, 1], [0, 0], 'k-', linewidth=2)
    
    # Отмечаем деления
    for i in range(11):
        x = i / 10
        h = 0.1 if i % 5 == 0 else 0.05
        ax.plot([x, x], [-h, h], 'k-', linewidth=1)
        if i % 5 == 0:
            ax.text(x, -0.2, f"{x:.1f}", fontsize=10, ha='center')
    
    # Отображаем вероятность
    cmap = plt.cm.Blues
    color = cmap(state.probability)
    ax.plot(state.probability, 0, 'o', markersize=15, color=color)
    ax.text(state.probability, 0.2, f"{state.probability:.4f}", fontsize=12, 
            ha='center', weight='bold')
    
    # Градиент фона для наглядности
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(gradient, cmap=cmap, aspect='auto', alpha=0.3,
              extent=[-0.1, 1.1, -0.5, 0.5])
    
    # Оформление
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def visualize_pbit(state: PBit, ax: Optional[plt.Axes] = None,
                  title: str = "P-бит") -> plt.Axes:
    """
    Визуализирует P-бит.
    
    Args:
        state: Объект PBit для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика.
        
    Returns:
        Объект осей matplotlib с визуализацией.
    """
    if not isinstance(state, PBit):
        raise TypeError(f"Expected PBit, got {type(state).__name__}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # Рисуем две шкалы: для положительной и отрицательной полярности
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    
    # Положительная ось
    ax.plot([0, 1], [1, 1], 'k-', linewidth=2)
    ax.text(-0.15, 1, "+1", fontsize=12, ha='center', va='center')
    
    # Отрицательная ось
    ax.plot([0, 1], [-1, -1], 'k-', linewidth=2)
    ax.text(-0.15, -1, "-1", fontsize=12, ha='center', va='center')
    
    # Отмечаем деления
    for i in range(11):
        x = i / 10
        h = 0.1
        ax.plot([x, x], [1-h, 1+h], 'k-', linewidth=1)
        ax.plot([x, x], [-1-h, -1+h], 'k-', linewidth=1)
        if i % 5 == 0:
            ax.text(x, 1-0.3, f"{x:.1f}", fontsize=10, ha='center')
            ax.text(x, -1-0.3, f"{x:.1f}", fontsize=10, ha='center')
    
    # Отображаем P-бит
    y_pos = 1 if state.sign == 1 else -1
    cmap = plt.cm.Blues if state.sign == 1 else plt.cm.Reds
    color = cmap(0.7)
    
    ax.plot(state.probability, y_pos, 'o', markersize=15, color=color)
    ax.text(state.probability, y_pos + 0.2 * state.sign, 
            f"({state.probability:.4f}, {state.sign:+d})", 
            fontsize=12, ha='center', weight='bold')
    
    # Оформление
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def visualize_phase_state(state: PhaseState, ax: Optional[plt.Axes] = None,
                        title: str = "Фазовое состояние") -> plt.Axes:
    """
    Визуализирует фазовое состояние.
    
    Args:
        state: Объект PhaseState для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика.
        
    Returns:
        Объект осей matplotlib с визуализацией.
    """
    if not isinstance(state, PhaseState):
        raise TypeError(f"Expected PhaseState, got {type(state).__name__}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Рисуем круг
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Координатные оси
    ax.plot([-1.2, 1.2], [0, 0], 'k--', alpha=0.5)
    ax.plot([0, 0], [-1.2, 1.2], 'k--', alpha=0.5)
    ax.text(1.2, 0, 'Re', fontsize=12, ha='left')
    ax.text(0, 1.2, 'Im', fontsize=12, va='bottom')
    
    # Отображаем фазовое состояние
    r = np.sqrt(state.probability)  # Радиус
    x = r * np.cos(state.phase)
    y = r * np.sin(state.phase)
    
    # Отображаем вектор состояния
    ax.plot([0, x], [0, y], '-', color='green', linewidth=2)
    ax.plot(x, y, 'o', markersize=10, color='green')
    
    # Отображаем угол
    if state.phase != 0:
        arc = Arc((0, 0), 0.5, 0.5, theta1=0, theta2=np.degrees(state.phase),
                 linewidth=2, color='blue', zorder=1)
        ax.add_patch(arc)
        ax.text(0.3 * np.cos(state.phase / 2), 0.3 * np.sin(state.phase / 2),
               f"{state.phase:.2f}", fontsize=10, ha='center', va='center',
               color='blue')
    
    # Подпись состояния
    margin = 0.05
    text_x = x + margin * np.cos(state.phase)
    text_y = y + margin * np.sin(state.phase)
    ax.text(text_x, text_y, f"({state.probability:.2f}, e^(i{state.phase:.2f}))",
           fontsize=12, ha='left' if x >= 0 else 'right')
    
    # Оформление
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def visualize_quantum_state(state: QuantumState, ax: Optional[plt.Axes] = None,
                          title: str = "Квантовое состояние") -> plt.Axes:
    """
    Визуализирует квантовое состояние на сфере Блоха.
    
    Args:
        state: Объект QuantumState для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика.
        
    Returns:
        Объект осей matplotlib с визуализацией.
    """
    if not isinstance(state, QuantumState):
        raise TypeError(f"Expected QuantumState, got {type(state).__name__}")
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Рисуем сферу Блоха
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='w', alpha=0.1)
    
    # Координатные оси
    ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'k--', alpha=0.5)
    ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'k--', alpha=0.5)
    ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'k--', alpha=0.5)
    
    ax.text(1.5, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.5, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.5, 'Z', fontsize=12)
    
    # Рисуем базисные состояния
    ax.plot([0, 0], [0, 0], [1, 1], 'o', markersize=10, color='blue')
    ax.text(0, 0, 1.1, '|0⟩', fontsize=12)
    ax.plot([0, 0], [0, 0], [-1, -1], 'o', markersize=10, color='red')
    ax.text(0, 0, -1.1, '|1⟩', fontsize=12)
    
    # Экватор
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 'g-', alpha=0.3)
    
    # Получаем координаты на сфере Блоха
    theta, phi = state.to_bloch()
    
    # Координаты точки на сфере
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Отображаем точку и вектор состояния
    ax.plot([0, x], [0, y], [0, z], 'purple', linewidth=2)
    ax.plot([x], [y], [z], 'o', markersize=10, color='purple')
    
    # Вычисляем амплитуды для подписи
    alpha, beta = state.amplitudes
    
    # Подпись состояния (упрощенная)
    ax.text(x, y, z + 0.1, f"|ψ⟩", fontsize=12, color='purple')
    
    # Оформление
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_title(title)
    
    # Удаляем метки для чистоты
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    return ax


def visualize_state(state: State, ax: Optional[plt.Axes] = None, 
                   title: Optional[str] = None) -> plt.Axes:
    """
    Визуализирует состояние любого уровня.
    
    Args:
        state: Объект State для визуализации.
        ax: Объект осей matplotlib (если None, создается новый).
        title: Заголовок графика (если None, генерируется автоматически).
        
    Returns:
        Объект осей matplotlib с визуализацией.
        
    Raises:
        TypeError: Если state не является объектом State.
    """
    if not isinstance(state, State):
        raise TypeError(f"Expected State, got {type(state).__name__}")
    
    # Определяем заголовок, если не указан
    if title is None:
        level_names = {
            1: "Классический бит",
            2: "Вероятностный бит",
            3: "P-бит",
            4: "Фазовое состояние",
            5: "Квантовое состояние"
        }
        title = f"{level_names.get(state.level, f'Уровень {state.level}')} - {state}"
    
    # Выбираем соответствующую функцию визуализации
    if isinstance(state, ClassicalBit):
        return visualize_classical_bit(state, ax, title)
    elif isinstance(state, ProbabilisticBit):
        return visualize_probabilistic_bit(state, ax, title)
    elif isinstance(state, PBit):
        return visualize_pbit(state, ax, title)
    elif isinstance(state, PhaseState):
        return visualize_phase_state(state, ax, title)
    elif isinstance(state, QuantumState):
        return visualize_quantum_state(state, ax, title)
    else:
        raise TypeError(f"Unsupported state type: {type(state).__name__}")


def visualize_operation(state1: State, state2: State, operation: str, result: State,
                       figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Визуализирует операцию между двумя состояниями.
    
    Args:
        state1: Первое состояние.
        state2: Второе состояние.
        operation: Строка с названием операции ('and', 'or', 'not').
        result: Результирующее состояние.
        figsize: Размер фигуры.
        
    Returns:
        Объект Figure matplotlib с визуализацией.
    """
    if operation == 'not':
        # Для операции NOT визуализируем только исходное и результирующее состояния
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        visualize_state(state1, axes[0], f"Исходное состояние\n{state1}")
        visualize_state(result, axes[1], f"NOT(Исходное)\n{result}")
    else:
        # Для бинарных операций визуализируем оба исходных состояния и результат
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        visualize_state(state1, axes[0], f"Состояние 1\n{state1}")
        visualize_state(state2, axes[1], f"Состояние 2\n{state2}")
        
        op_symbol = {'and': '⊗', 'or': '⊕'}.get(operation, operation)
        op_name = {'and': 'AND', 'or': 'OR'}.get(operation, operation.upper())
        visualize_state(result, axes[2], f"Результат {op_name}\n{state1} {op_symbol} {state2} = {result}")
    
    fig.tight_layout()
    return fig


def visualize_lifting(state: State, lifted_state: State, 
                    figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Визуализирует подъем состояния на более высокий уровень.
    
    Args:
        state: Исходное состояние.
        lifted_state: Поднятое состояние.
        figsize: Размер фигуры.
        
    Returns:
        Объект Figure matplotlib с визуализацией.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Визуализируем исходное и результирующее состояния
    visualize_state(state, axes[0], f"Исходное состояние (Уровень {state.level})\n{state}")
    visualize_state(lifted_state, axes[1], f"Поднятое состояние (Уровень {lifted_state.level})\n{lifted_state}")
    
    # Добавляем заголовок с описанием операции
    fig.suptitle(f"Операция подъема L{state.level}→{lifted_state.level}", fontsize=16)
    
    fig.tight_layout()
    return fig


def visualize_projection(state: State, projected_state: State, 
                       figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Визуализирует проекцию состояния на более низкий уровень.
    
    Args:
        state: Исходное состояние.
        projected_state: Проецированное состояние.
        figsize: Размер фигуры.
        
    Returns:
        Объект Figure matplotlib с визуализацией.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Визуализируем исходное и результирующее состояния
    visualize_state(state, axes[0], f"Исходное состояние (Уровень {state.level})\n{state}")
    visualize_state(projected_state, axes[1], f"Проецированное состояние (Уровень {projected_state.level})\n{projected_state}")
    
    # Добавляем заголовок с описанием операции
    fig.suptitle(f"Операция проекции P{state.level}→{projected_state.level}", fontsize=16)
    
    fig.tight_layout()
    return fig
