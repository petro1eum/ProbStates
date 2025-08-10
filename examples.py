# examples.py
"""
Примеры использования библиотеки probstates для работы с формализмом
вероятностных состояний.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D support
from probstates import (
    ClassicalBit, 
    ProbabilisticBit, 
    PBit, 
    PhaseState, 
    QuantumState,
    lift,
    project,
    PhaseRegister,
    deutsch_jozsa,
)
from probstates.visualization import (
    visualize_state,
    visualize_operation,
    visualize_lifting,
    visualize_projection
)


def basic_states_example():
    """Демонстрирует создание и визуализацию состояний всех уровней."""
    # Создаем состояния каждого уровня
    c_bit = ClassicalBit(1)
    p_bit = ProbabilisticBit(0.7)
    pb_bit = PBit(0.6, -1)
    ph_state = PhaseState(0.8, np.pi/4)
    q_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])
    
    # Визуализируем каждое состояние
    fig = plt.figure(figsize=(10, 25))
    
    # Создаем отдельные подграфики для каждого состояния
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    # Для квантового состояния нужен 3D-подграфик
    ax5 = fig.add_subplot(5, 1, 5, projection='3d')
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    visualize_state(c_bit, axes[0], "Классический бит")
    visualize_state(p_bit, axes[1], "Вероятностный бит")
    visualize_state(pb_bit, axes[2], "P-бит")
    visualize_state(ph_state, axes[3], "Фазовое состояние")
    visualize_state(q_state, axes[4], "Квантовое состояние")
    
    fig.tight_layout()
    plt.savefig("basic_states.png")
    plt.close(fig)
    print("Базовые состояния созданы и визуализированы в файле basic_states.png")


def operations_example():
    """Демонстрирует операции на каждом уровне."""
    # Операции на уровне 1 (классические биты)
    print("\n=== Операции на уровне 1 (классические биты) ===")
    bit1 = ClassicalBit(0)
    bit2 = ClassicalBit(1)
    
    print(f"{bit1} AND {bit2} = {bit1 & bit2}")
    print(f"{bit1} OR {bit2} = {bit1 | bit2}")
    print(f"NOT {bit1} = {~bit1}")
    
    # Операции на уровне 2 (вероятностные биты)
    print("\n=== Операции на уровне 2 (вероятностные биты) ===")
    prob1 = ProbabilisticBit(0.3)
    prob2 = ProbabilisticBit(0.6)
    
    print(f"{prob1} AND {prob2} = {prob1 & prob2}")
    print(f"{prob1} OR {prob2} = {prob1 | prob2}")
    print(f"NOT {prob1} = {~prob1}")
    
    # Операции на уровне 3 (P-биты)
    print("\n=== Операции на уровне 3 (P-биты) ===")
    pbit1 = PBit(0.3, +1)
    pbit2 = PBit(0.6, -1)
    
    print(f"{pbit1} AND {pbit2} = {pbit1 & pbit2}")
    print(f"{pbit1} OR {pbit2} = {pbit1 | pbit2}")
    print(f"NOT {pbit1} = {~pbit1}")
    
    # Операции на уровне 4 (фазовые состояния)
    print("\n=== Операции на уровне 4 (фазовые состояния) ===")
    phase1 = PhaseState(0.3, 0)
    phase2 = PhaseState(0.6, np.pi/2)
    
    print(f"{phase1} AND {phase2} = {phase1 & phase2}")
    print(f"{phase1} OR {phase2} = {phase1 | phase2}")
    print(f"NOT {phase1} = {~phase1}")
    
    # Визуализируем операцию OR на фазовых состояниях
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    visualize_state(phase1, ax1, f"Состояние 1\n{phase1}")
    visualize_state(phase2, ax2, f"Состояние 2\n{phase2}")
    
    op_symbol = {'and': '⊗', 'or': '⊕'}.get('or', 'or')
    op_name = {'and': 'AND', 'or': 'OR'}.get('or', 'or'.upper())
    visualize_state(phase1 | phase2, ax3, f"Результат {op_name}\n{phase1} {op_symbol} {phase2} = {phase1 | phase2}")
    
    fig.tight_layout()
    plt.savefig("phase_or_operation.png")
    plt.close(fig)
    print("Операция OR на фазовых состояниях визуализирована в файле phase_or_operation.png")
    
    # Операции на уровне 5 (квантовые состояния)
    print("\n=== Операции на уровне 5 (квантовые состояния) ===")
    # Создаем два ортогональных состояния
    qubit1 = QuantumState([1, 0])  # |0⟩
    qubit2 = QuantumState([0, 1])  # |1⟩
    
    print(f"{qubit1} AND {qubit2} = {qubit1 & qubit2}")
    print(f"{qubit1} OR {qubit2} = {qubit1 | qubit2}")
    print(f"NOT {qubit1} = {~qubit1}")
    
    # Визуализируем операцию OR на квантовых состояниях
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    visualize_state(qubit1, ax1, f"Состояние 1\n{qubit1}")
    visualize_state(qubit2, ax2, f"Состояние 2\n{qubit2}")
    
    op_symbol = {'and': '⊗', 'or': '⊕'}.get('or', 'or')
    op_name = {'and': 'AND', 'or': 'OR'}.get('or', 'or'.upper())
    visualize_state(qubit1 | qubit2, ax3, f"Результат {op_name}\n{qubit1} {op_symbol} {qubit2} = {qubit1 | qubit2}")
    
    fig.tight_layout()
    plt.savefig("quantum_or_operation.png")
    plt.close(fig)
    print("Операция OR на квантовых состояниях визуализирована в файле quantum_or_operation.png")


def lifting_example():
    """Демонстрирует операцию подъема состояний."""
    print("\n=== Операции подъема (Lifting) ===")
    
    # Начинаем с классического бита
    bit = ClassicalBit(1)
    print(f"Исходное состояние: {bit}")
    
    # Поднимаем до вероятностного бита
    prob_bit = lift(bit, 2)
    print(f"Уровень 1 → 2: {bit} → {prob_bit}")
    
    # Поднимаем до P-бита
    p_bit = lift(prob_bit, 3)
    print(f"Уровень 2 → 3: {prob_bit} → {p_bit}")
    
    # Поднимаем до фазового состояния
    phase_state = lift(p_bit, 4)
    print(f"Уровень 3 → 4: {p_bit} → {phase_state}")
    
    # Поднимаем до квантового состояния
    quantum_state = lift(phase_state, 5)
    print(f"Уровень 4 → 5: {phase_state} → {quantum_state}")
    
    # Визуализируем цепочку подъема
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 5, 1)
    ax2 = fig.add_subplot(1, 5, 2)
    ax3 = fig.add_subplot(1, 5, 3)
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5, projection='3d')
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    visualize_state(bit, axes[0], "Уровень 1")
    visualize_state(prob_bit, axes[1], "Уровень 2")
    visualize_state(p_bit, axes[2], "Уровень 3")
    visualize_state(phase_state, axes[3], "Уровень 4")
    visualize_state(quantum_state, axes[4], "Уровень 5")
    
    fig.suptitle("Цепочка подъема состояний", fontsize=16)
    fig.tight_layout()
    plt.savefig("lifting_chain.png")
    plt.close(fig)
    print("Цепочка подъема визуализирована в файле lifting_chain.png")
    
    # Визуализация конкретного шага подъема
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    visualize_state(p_bit, ax1, "Исходное состояние (Уровень 3)")
    visualize_state(phase_state, ax2, "Поднятое состояние (Уровень 4)")
    
    fig.suptitle(f"Операция подъема L{p_bit.level}→{phase_state.level}", fontsize=16)
    fig.tight_layout()
    plt.savefig("lifting_step.png")
    plt.close(fig)
    print("Шаг подъема визуализирован в файле lifting_step.png")


def projection_example():
    """Демонстрирует операцию проекции состояний."""
    print("\n=== Операции проекции (Projection) ===")
    
    # Начинаем с квантового состояния (равная суперпозиция)
    quantum_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])
    print(f"Исходное состояние: {quantum_state}")
    
    # Проецируем на фазовое состояние
    phase_state = project(quantum_state, 4)
    print(f"Уровень 5 → 4: {quantum_state} → {phase_state}")
    
    # Проецируем на P-бит
    p_bit = project(phase_state, 3)
    print(f"Уровень 4 → 3: {phase_state} → {p_bit}")
    
    # Проецируем на вероятностный бит
    prob_bit = project(p_bit, 2)
    print(f"Уровень 3 → 2: {p_bit} → {prob_bit}")
    
    # Проецируем на классический бит
    bit = project(prob_bit, 1)
    print(f"Уровень 2 → 1: {prob_bit} → {bit}")
    
    # Визуализируем цепочку проекции
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 5, 1, projection='3d')
    ax2 = fig.add_subplot(1, 5, 2)
    ax3 = fig.add_subplot(1, 5, 3)
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5)
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    visualize_state(quantum_state, axes[0], "Уровень 5")
    visualize_state(phase_state, axes[1], "Уровень 4")
    visualize_state(p_bit, axes[2], "Уровень 3")
    visualize_state(prob_bit, axes[3], "Уровень 2")
    visualize_state(bit, axes[4], "Уровень 1")
    
    fig.suptitle("Цепочка проекции состояний", fontsize=16)
    fig.tight_layout()
    plt.savefig("projection_chain.png")
    plt.close(fig)
    print("Цепочка проекции визуализирована в файле projection_chain.png")
    
    # Визуализация конкретного шага проекции
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    visualize_state(phase_state, ax1, "Исходное состояние (Уровень 4)")
    visualize_state(p_bit, ax2, "Проецированное состояние (Уровень 3)")
    
    fig.suptitle(f"Операция проекции P{phase_state.level}→{p_bit.level}", fontsize=16)
    fig.tight_layout()
    plt.savefig("projection_step.png")
    plt.close(fig)
    print("Шаг проекции визуализирован в файле projection_step.png")


def coin_flip_example():
    """
    Моделирует подбрасывание монеты на разных уровнях.
    """
    print("\n=== Моделирование подбрасывания монеты ===")
    
    # Фиксируем seed для воспроизводимости
    rng = np.random.RandomState(42)
    
    # Уровень 1: Классический бит (детерминированный)
    c_bit = ClassicalBit(1)  # Всегда орел
    print(f"Уровень 1 (детерминированный): {c_bit}")
    
    # Уровень 2: Вероятностный бит (случайный)
    p_bit = ProbabilisticBit(0.5)  # Честная монета
    
    # Генерируем 10 результатов подбрасывания
    results = [p_bit.sample(rng) for _ in range(10)]
    print(f"Уровень 2 (вероятностный): 10 подбрасываний: {results}")
    
    # Уровень 3: P-бит (с полярностью)
    pb_bit = PBit(0.5, -1)  # Честная монета с отрицательной полярностью
    
    # Генерируем 5 результатов подбрасывания
    results = [pb_bit.sample(rng) for _ in range(5)]
    print(f"Уровень 3 (P-бит): 5 подбрасываний: {results}")
    
    # Уровень 4: Фазовое состояние
    ph_state = PhaseState(0.5, np.pi/4)  # Честная монета с фазой π/4
    
    # Генерируем 5 результатов подбрасывания
    results = [ph_state.sample(rng) for _ in range(5)]
    print(f"Уровень 4 (фазовое): 5 подбрасываний: {results}")
    
    # Уровень 5: Квантовое состояние
    q_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])  # Равная суперпозиция
    
    # Генерируем 10 результатов измерения
    results = [q_state.measure(rng) for _ in range(10)]
    print(f"Уровень 5 (квантовое): 10 измерений: {results}")
    
    # Визуализируем все состояния
    fig = plt.figure(figsize=(10, 20))
    
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5, projection='3d')
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    visualize_state(c_bit, axes[0], "Уровень 1: Детерминированная монета")
    visualize_state(p_bit, axes[1], "Уровень 2: Вероятностная монета")
    visualize_state(pb_bit, axes[2], "Уровень 3: Монета с полярностью")
    visualize_state(ph_state, axes[3], "Уровень 4: Монета с фазой")
    visualize_state(q_state, axes[4], "Уровень 5: Квантовая монета")
    
    fig.suptitle("Моделирование подбрасывания монеты на разных уровнях", fontsize=16)
    fig.tight_layout()
    plt.savefig("coin_flip_models.png")
    plt.close(fig)
    print("Модели подбрасывания монеты визуализированы в файле coin_flip_models.png")


def interference_example():
    """
    Демонстрирует интерференцию на уровнях 3, 4 и 5.
    """
    print("\n=== Демонстрация интерференции ===")
    
    # Уровень 3: P-биты
    # Создаем два P-бита с противоположными полярностями
    pb1 = PBit(0.5, +1)
    pb2 = PBit(0.5, -1)
    
    # Складываем их (интерференция)
    pb_result = pb1 | pb2
    
    print(f"Уровень 3: ({pb1}) ⊕ ({pb2}) = {pb_result}")
    print(f"  Демонстрирует деструктивную интерференцию P-битов")
    
    # Уровень 4: Фазовые состояния
    # Создаем два фазовых состояния с противоположными фазами
    ph1 = PhaseState(0.5, 0)      # Фаза 0
    ph2 = PhaseState(0.5, np.pi)  # Фаза π (противоположная)
    
    # Складываем их (интерференция)
    ph_result = ph1 | ph2
    
    print(f"Уровень 4: ({ph1}) ⊕ ({ph2}) = {ph_result}")
    print(f"  Демонстрирует деструктивную интерференцию фазовых состояний")
    
    # Уровень 5: Квантовые состояния
    # Создаем два квантовых состояния в противофазе
    q1 = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])    # |+⟩ = (|0⟩ + |1⟩)/√2
    q2 = QuantumState([1/np.sqrt(2), -1/np.sqrt(2)])   # |-⟩ = (|0⟩ - |1⟩)/√2
    
    # Складываем их (интерференция)
    q_result = q1 | q2
    
    print(f"Уровень 5: ({q1}) ⊕ ({q2}) ≈ {q_result}")
    print(f"  Демонстрирует интерференцию квантовых состояний")
    
    # Визуализируем интерференцию
    # P-биты
    fig1 = plt.figure(figsize=(15, 5))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax2 = fig1.add_subplot(1, 3, 2)
    ax3 = fig1.add_subplot(1, 3, 3)
    
    visualize_state(pb1, ax1)
    visualize_state(pb2, ax2)
    visualize_state(pb_result, ax3)
    
    fig1.suptitle("Интерференция P-битов (Уровень 3)", fontsize=16)
    fig1.tight_layout()
    plt.savefig("pbit_interference.png")
    plt.close(fig1)
    
    # Фазовые состояния
    fig2 = plt.figure(figsize=(15, 5))
    ax1 = fig2.add_subplot(1, 3, 1)
    ax2 = fig2.add_subplot(1, 3, 2)
    ax3 = fig2.add_subplot(1, 3, 3)
    
    visualize_state(ph1, ax1)
    visualize_state(ph2, ax2)
    visualize_state(ph_result, ax3)
    
    fig2.suptitle("Интерференция фазовых состояний (Уровень 4)", fontsize=16)
    fig2.tight_layout()
    plt.savefig("phase_interference.png")
    plt.close(fig2)
    
    # Квантовые состояния
    fig3 = plt.figure(figsize=(15, 5))
    ax1 = fig3.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig3.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig3.add_subplot(1, 3, 3, projection='3d')
    
    visualize_state(q1, ax1)
    visualize_state(q2, ax2)
    visualize_state(q_result, ax3)
    
    fig3.suptitle("Интерференция квантовых состояний (Уровень 5)", fontsize=16)
    fig3.tight_layout()
    plt.savefig("quantum_interference.png")
    plt.close(fig3)
    
    print("Интерференция визуализирована в файлах pbit_interference.png, phase_interference.png и quantum_interference.png")


def deutsch_jozsa_example():
    """
    Демонстрирует работу алгоритма Дойча–Йожи на фазовом регистре.
    """
    print("\n=== Алгоритм Дойча–Йожи (фазовый регистр) ===")
    n = 3

    # Константные оракулы
    def f_const0(x: int) -> int:
        return 0

    def f_const1(x: int) -> int:
        return 1

    # Сбалансированные оракулы (ровно половина входов -> 1)
    def f_parity(x: int) -> int:
        # Чётность битов x
        return bin(x).count("1") & 1

    def f_msb(x: int) -> int:
        # Старший бит равен 1 для половины входов
        return (x >> (n - 1)) & 1

    for name, f in [
        ("const-0", f_const0),
        ("const-1", f_const1),
        ("parity", f_parity),
        ("msb", f_msb),
    ]:
        kind, p0 = deutsch_jozsa(f, n)
        print(f"oracle={name:7s} → predicted={kind:9s}, p0={p0:.12f}")


if __name__ == "__main__":
    # Запускаем все примеры
    print("=== Запуск примеров использования библиотеки probstates ===")
    
    basic_states_example()
    operations_example()
    lifting_example()
    projection_example()
    coin_flip_example()
    interference_example()
    
    print("\n=== Все примеры выполнены успешно ===")
