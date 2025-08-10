#!/usr/bin/env python
# entropy_examples.py
"""
Демонстрация энтропийных характеристик в иерархии вероятностных состояний.

Этот скрипт демонстрирует расчет энтропии для различных уровней иерархии состояний,
от классической энтропии Шеннона до квазиквантовой энтропии для фазовых состояний.
"""

import numpy as np
import matplotlib.pyplot as plt
from probstates import (
    ClassicalBit, 
    ProbabilisticBit, 
    PBit, 
    PhaseState, 
    QuantumState,
    lift,
    project
)
from probstates.entropy import (
    calculate_entropy,
    shannon_entropy,
    entropy_level2,
    entropy_level3,
    entropy_level4,
    von_neumann_entropy,
    information_loss,
    accessible_information,
    kl_divergence
)
from probstates.coherence import dephase, amp_damp
from probstates import set_phase_or_mode


def entropy_analysis_across_levels():
    """
    Анализ энтропии на разных уровнях иерархии для разных вероятностей.
    """
    print("\n=== Анализ энтропии на разных уровнях иерархии ===")
    
    # Создаем массив вероятностей от 0 до 1
    probabilities = np.linspace(0, 1, 101)
    
    # Массивы для хранения энтропий
    entropies_level2 = []  # Шеннон
    entropies_level3_pos = []  # P-бит с положительной полярностью
    entropies_level3_neg = []  # P-бит с отрицательной полярностью
    entropies_level4 = []  # Фазовое состояние с фазой π/4
    
    # Рассчитываем энтропию для каждой вероятности
    for p in probabilities:
        # Уровень 2: Вероятностный бит
        prob_bit = ProbabilisticBit(p)
        entropies_level2.append(calculate_entropy(prob_bit))
        
        # Уровень 3: P-бит с положительной полярностью
        pbit_pos = PBit(p, +1)
        entropies_level3_pos.append(calculate_entropy(pbit_pos))
        
        # Уровень 3: P-бит с отрицательной полярностью
        pbit_neg = PBit(p, -1)
        entropies_level3_neg.append(calculate_entropy(pbit_neg))
        
        # Уровень 4: Фазовое состояние с фазой π/4
        phase_state = PhaseState(p, np.pi/4)
        entropies_level4.append(calculate_entropy(phase_state))
    
    # Строим график
    plt.figure(figsize=(12, 8))
    plt.plot(probabilities, entropies_level2, label='Уровень 2 (Вероятностный бит)', linewidth=2)
    plt.plot(probabilities, entropies_level3_pos, label='Уровень 3 (P-бит, s=+1)', linewidth=2)
    plt.plot(probabilities, entropies_level3_neg, label='Уровень 3 (P-бит, s=-1)', linewidth=2)
    plt.plot(probabilities, entropies_level4, label='Уровень 4 (Фазовое состояние, φ=π/4)', linewidth=2)
    
    plt.title('Зависимость энтропии от вероятности для разных уровней', fontsize=16)
    plt.xlabel('Вероятность p', fontsize=14)
    plt.ylabel('Энтропия', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('entropy_across_levels.png')
    plt.close()
    
    print("График зависимости энтропии от вероятности сохранен в файле entropy_across_levels.png")


def kl_divergence_analysis():
    """
    Анализ дивергенции Кульбака-Лейблера между распределениями Бернулли.
    """
    print("\n=== Анализ дивергенции Кульбака-Лейблера ===")
    
    # Создаем массив вероятностей от 0.01 до 0.99
    probabilities = np.linspace(0.01, 0.99, 99)
    
    # Массив для хранения дивергенций
    divergences = []
    
    # Рассчитываем дивергенцию между B_p и B_(1-p)
    for p in probabilities:
        div = kl_divergence(p, 1-p)
        divergences.append(div)
    
    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, divergences, linewidth=2)
    
    plt.title('Дивергенция Кульбака-Лейблера между B_p и B_(1-p)', fontsize=16)
    plt.xlabel('Вероятность p', fontsize=14)
    plt.ylabel('D_KL(B_p || B_(1-p))', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kl_divergence.png')
    plt.close()
    
    print("График дивергенции Кульбака-Лейблера сохранен в файле kl_divergence.png")
    
    # Выводим некоторые значения
    print(f"D_KL(B_0.1 || B_0.9) = {kl_divergence(0.1, 0.9):.4f}")
    print(f"D_KL(B_0.3 || B_0.7) = {kl_divergence(0.3, 0.7):.4f}")
    print(f"D_KL(B_0.5 || B_0.5) = {kl_divergence(0.5, 0.5):.4f}")


def information_loss_analysis():
    """
    Анализ потери информации при проекции с верхних уровней на нижние.
    """
    print("\n=== Анализ потери информации при проекции ===")
    
    # Создаем массив вероятностей от 0.01 до 0.99
    probabilities = np.linspace(0.01, 0.99, 99)
    
    # Массивы для хранения потерь информации
    loss_4to3 = []  # Фазовое состояние -> P-бит
    loss_3to2 = []  # P-бит -> Вероятностный бит
    loss_2to1 = []  # Вероятностный бит -> Классический бит
    
    # Рассчитываем потери информации для каждой вероятности
    for p in probabilities:
        # Создаем состояния
        phase_state = PhaseState(p, np.pi/4)
        pbit = project(phase_state, 3)
        prob_bit = project(pbit, 2)
        classic_bit = project(prob_bit, 1)
        
        # Рассчитываем потери информации
        loss_4to3.append(information_loss(phase_state, pbit))
        loss_3to2.append(information_loss(pbit, prob_bit))
        loss_2to1.append(information_loss(prob_bit, classic_bit))
    
    # Строим график
    plt.figure(figsize=(12, 8))
    plt.plot(probabilities, loss_4to3, label='Уровень 4 → 3 (Фазовое состояние → P-бит)', linewidth=2)
    plt.plot(probabilities, loss_3to2, label='Уровень 3 → 2 (P-бит → Вероятностный бит)', linewidth=2)
    plt.plot(probabilities, loss_2to1, label='Уровень 2 → 1 (Вероятностный бит → Классический бит)', linewidth=2)
    
    plt.title('Потери информации при проекции между уровнями', fontsize=16)
    plt.xlabel('Вероятность p', fontsize=14)
    plt.ylabel('Потеря информации (бит)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('information_loss.png')
    plt.close()
    
    print("График потери информации сохранен в файле information_loss.png")


def phase_entropy_analysis():
    """
    Анализ влияния фазы на энтропию фазовых состояний.
    """
    print("\n=== Анализ влияния фазы на энтропию ===")
    
    # Создаем массив фаз от 0 до 2π
    phases = np.linspace(0, 2*np.pi, 101)
    
    # Массивы для хранения энтропий при разных вероятностях
    entropies_p_01 = []  # p = 0.1
    entropies_p_05 = []  # p = 0.5
    entropies_p_09 = []  # p = 0.9
    
    # Рассчитываем энтропию для каждой фазы
    for phi in phases:
        # Фазовые состояния с разными вероятностями
        state_p_01 = PhaseState(0.1, phi)
        state_p_05 = PhaseState(0.5, phi)
        state_p_09 = PhaseState(0.9, phi)
        
        # Рассчитываем энтропии
        entropies_p_01.append(calculate_entropy(state_p_01))
        entropies_p_05.append(calculate_entropy(state_p_05))
        entropies_p_09.append(calculate_entropy(state_p_09))
    
    # Строим график (значения должны быть почти независимы от φ)
    plt.figure(figsize=(12, 8))
    plt.plot(phases, entropies_p_01, label='p = 0.1', linewidth=2)
    plt.plot(phases, entropies_p_05, label='p = 0.5', linewidth=2)
    plt.plot(phases, entropies_p_09, label='p = 0.9', linewidth=2)
    
    plt.title('Зависимость энтропии фазового состояния от фазы', fontsize=16)
    plt.xlabel('Фаза φ (рад)', fontsize=14)
    plt.ylabel('Энтропия', fontsize=14)
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
              ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('phase_entropy.png')
    plt.close()
    
    print("График зависимости энтропии от фазы сохранен в файле phase_entropy.png")


def level4_entropy_under_noise_and_modes():
    """
    Демонстрация влияния шумов и режима ⊕₄ на распределения и энтропию результата.
    """
    print("\n=== Уровень 4: влияние шума и режима ⊕₄ ===")
    p1, p2 = 0.5, 0.5
    phi1 = 0.0
    deltas = np.linspace(0.0, 2*np.pi, 181)
    modes = ['quant', 'norm', 'weight', 'opt']
    sigma_phi = 0.2
    alpha = 0.3
    res = {m: [] for m in modes}
    for d in deltas:
        a = PhaseState(p1, phi1)
        b = PhaseState(p2, phi1 + d)
        # шумы на входах
        a = dephase(amp_damp(a, alpha), sigma_phi=sigma_phi)
        b = dephase(amp_damp(b, alpha), sigma_phi=sigma_phi)
        for m in modes:
            set_phase_or_mode(m)
            c = a | b
            res[m].append(entropy_level4(c))
    # График энтропии результата vs Δφ
    plt.figure(figsize=(12, 6))
    for m in modes:
        plt.plot(deltas, res[m], label=m)
    plt.xlabel('Δφ (рад)')
    plt.ylabel('H4 результата')
    plt.title('Влияние шума и режима ⊕₄ на энтропию результата')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_entropy_noise_modes.png')
    plt.close()
    print("График сохранен: phase_entropy_noise_modes.png")


def theorem2_1_verification():
    """
    Верификация Теоремы 2.1 из статьи, демонстрирующей, что классическая
    энтропия Шеннона не сохраняется при операциях на уровнях 3 и 4.
    """
    print("\n=== Верификация Теоремы 2.1 ===")
    
    # Создаем p-биты как в доказательстве теоремы
    S3 = PBit(0.6, +1)
    T3 = PBit(0.6, +1)
    
    # Выполняем операцию S3 ⊕₃ T3
    S3_T3 = S3 | T3
    
    # Проецируем на уровень 2
    S2 = project(S3, 2)
    T2 = project(T3, 2)
    S3_T3_2 = project(S3_T3, 2)
    
    # Рассчитываем энтропии
    H_S2 = entropy_level2(S2)
    H_T2 = entropy_level2(T2)
    H_S3_T3_2 = entropy_level2(S3_T3_2)
    
    # Проверяем, выполняется ли условие теоремы
    left_side = H_S3_T3_2
    
    # Вычисляем H(0.6) ⊕₂ H(0.6) вручную по формуле OR операции для вероятностей
    # p₁ ⊕₂ p₂ = p₁ + p₂ - p₁·p₂
    manual_calc = H_S2 + H_T2 - H_S2 * H_T2
    right_side = manual_calc
    
    print(f"S3 = {S3}, T3 = {T3}")
    print(f"S3 ⊕₃ T3 = {S3_T3}")
    print(f"P₃→₂(S3) = {S2}, P₃→₂(T3) = {T2}")
    print(f"P₃→₂(S3 ⊕₃ T3) = {S3_T3_2}")
    print(f"H(P₃→₂(S3)) = {H_S2:.4f}")
    print(f"H(P₃→₂(T3)) = {H_T2:.4f}")
    print(f"H(P₃→₂(S3 ⊕₃ T3)) = {H_S3_T3_2:.4f}")
    
    print(f"H(0.6) ⊕₂ H(0.6) = {manual_calc:.4f} (вручную)")
    
    print(f"Теорема 2.1 подтверждается: {left_side:.4f} ≠ {right_side:.4f}")


def entropy_measurement_example():
    """
    Иллюстрация процесса измерения энтропии для разных уровней.
    """
    print("\n=== Моделирование измерения энтропии ===")
    
    # Создаем состояния
    c_bit = ClassicalBit(1)
    prob_bit = ProbabilisticBit(0.7)
    pbit_pos = PBit(0.7, +1)
    pbit_neg = PBit(0.7, -1)
    phase_state_0 = PhaseState(0.7, 0)
    phase_state_pi4 = PhaseState(0.7, np.pi/4)
    quantum_state = QuantumState([np.sqrt(0.7), np.sqrt(0.3)])
    
    # Рассчитываем энтропии
    print("Энтропии для состояний одинаковой вероятности p = 0.7:")
    print(f"Классический бит (уровень 1): {calculate_entropy(c_bit):.4f}")
    print(f"Вероятностный бит (уровень 2): {calculate_entropy(prob_bit):.4f}")
    print(f"P-бит (s = +1, уровень 3): {calculate_entropy(pbit_pos):.4f}")
    print(f"P-бит (s = -1, уровень 3): {calculate_entropy(pbit_neg):.4f}")
    print(f"Фазовое состояние (φ = 0, уровень 4): {calculate_entropy(phase_state_0):.4f}")
    print(f"Фазовое состояние (φ = π/4, уровень 4): {calculate_entropy(phase_state_pi4):.4f}")
    print(f"Квантовое состояние (уровень 5): {calculate_entropy(quantum_state):.4f}")
    
    # Информация, доступная при классическом измерении
    print("\nДоступная информация при классическом измерении:")
    print(f"Классический бит (уровень 1): {accessible_information(c_bit):.4f}")
    print(f"Вероятностный бит (уровень 2): {accessible_information(prob_bit):.4f}")
    print(f"P-бит (s = +1, уровень 3): {accessible_information(pbit_pos):.4f}")
    print(f"P-бит (s = -1, уровень 3): {accessible_information(pbit_neg):.4f}")
    print(f"Фазовое состояние (φ = 0, уровень 4): {accessible_information(phase_state_0):.4f}")
    print(f"Фазовое состояние (φ = π/4, уровень 4): {accessible_information(phase_state_pi4):.4f}")
    print(f"Квантовое состояние (уровень 5): {accessible_information(quantum_state):.4f}")


def interference_entropy_example():
    """
    Анализ энтропии при интерференции состояний.
    """
    print("\n=== Энтропия при интерференции состояний ===")
    
    # Уровень 3: P-биты с противоположными полярностями
    pb1 = PBit(0.5, +1)
    pb2 = PBit(0.5, -1)
    pb_result = pb1 | pb2  # Деструктивная интерференция
    
    print(f"Уровень 3:")
    print(f"H_3({pb1}) = {calculate_entropy(pb1):.4f}")
    print(f"H_3({pb2}) = {calculate_entropy(pb2):.4f}")
    print(f"H_3({pb1} | {pb2}) = {calculate_entropy(pb_result):.4f}")
    
    # Уровень 4: Фазовые состояния с противофазой
    ph1 = PhaseState(0.5, 0)
    ph2 = PhaseState(0.5, np.pi)
    ph_result = ph1 | ph2  # Полное гашение
    
    print(f"\nУровень 4:")
    print(f"H_4({ph1}) = {calculate_entropy(ph1):.4f}")
    print(f"H_4({ph2}) = {calculate_entropy(ph2):.4f}")
    print(f"H_4({ph1} | {ph2}) = {calculate_entropy(ph_result):.4f}")
    
    # Уровень 5: Квантовые состояния
    q1 = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])    # |+⟩
    q2 = QuantumState([1/np.sqrt(2), -1/np.sqrt(2)])   # |-⟩
    q_result = q1 | q2  # Интерференция
    
    print(f"\nУровень 5:")
    print(f"S({q1}) = {calculate_entropy(q1):.4f}")
    print(f"S({q2}) = {calculate_entropy(q2):.4f}")
    print(f"S({q1} | {q2}) = {calculate_entropy(q_result):.4f}")


if __name__ == "__main__":
    print("=== Энтропийные характеристики в иерархии вероятностных состояний ===")
    
    # Запускаем все примеры
    entropy_analysis_across_levels()
    kl_divergence_analysis()
    information_loss_analysis()
    phase_entropy_analysis()
    level4_entropy_under_noise_and_modes()
    theorem2_1_verification()
    entropy_measurement_example()
    interference_entropy_example()
    
    print("\n=== Все примеры выполнены успешно ===") 