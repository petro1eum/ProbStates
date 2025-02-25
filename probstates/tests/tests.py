# tests/test_basic.py
"""
Базовые тесты для библиотеки probstates.
"""

import unittest
import numpy as np
from probstates import (
    ClassicalBit, 
    ProbabilisticBit, 
    PBit, 
    PhaseState, 
    QuantumState,
    lift,
    project
)


class TestClassicalBit(unittest.TestCase):
    """Тесты для класса ClassicalBit."""
    
    def test_creation(self):
        """Тест создания классического бита."""
        bit0 = ClassicalBit(0)
        bit1 = ClassicalBit(1)
        
        self.assertEqual(bit0.value, 0)
        self.assertEqual(bit1.value, 1)
        
        # Проверка некорректных значений
        with self.assertRaises(ValueError):
            ClassicalBit(2)
        
        # Проверка создания из bool
        self.assertEqual(ClassicalBit(True).value, 1)
        self.assertEqual(ClassicalBit(False).value, 0)
    
    def test_operations(self):
        """Тест операций над классическими битами."""
        bit0 = ClassicalBit(0)
        bit1 = ClassicalBit(1)
        
        # Операция AND
        self.assertEqual((bit0 & bit0).value, 0)
        self.assertEqual((bit0 & bit1).value, 0)
        self.assertEqual((bit1 & bit0).value, 0)
        self.assertEqual((bit1 & bit1).value, 1)
        
        # Операция OR
        self.assertEqual((bit0 | bit0).value, 0)
        self.assertEqual((bit0 | bit1).value, 1)
        self.assertEqual((bit1 | bit0).value, 1)
        self.assertEqual((bit1 | bit1).value, 1)
        
        # Операция NOT
        self.assertEqual((~bit0).value, 1)
        self.assertEqual((~bit1).value, 0)
    
    def test_equality(self):
        """Тест сравнения классических битов."""
        bit0 = ClassicalBit(0)
        bit1 = ClassicalBit(1)
        
        self.assertEqual(bit0, ClassicalBit(0))
        self.assertEqual(bit1, ClassicalBit(1))
        self.assertNotEqual(bit0, bit1)
        
        # Сравнение с числами и bool
        self.assertEqual(bit0, 0)
        self.assertEqual(bit1, 1)
        self.assertEqual(bit0, False)
        self.assertEqual(bit1, True)


class TestProbabilisticBit(unittest.TestCase):
    """Тесты для класса ProbabilisticBit."""
    
    def test_creation(self):
        """Тест создания вероятностного бита."""
        p0 = ProbabilisticBit(0)
        p1 = ProbabilisticBit(1)
        p05 = ProbabilisticBit(0.5)
        
        self.assertEqual(p0.probability, 0)
        self.assertEqual(p1.probability, 1)
        self.assertEqual(p05.probability, 0.5)
        
        # Проверка некорректных значений
        with self.assertRaises(ValueError):
            ProbabilisticBit(-0.1)
        with self.assertRaises(ValueError):
            ProbabilisticBit(1.1)
    
    def test_operations(self):
        """Тест операций над вероятностными битами."""
        p0 = ProbabilisticBit(0)
        p1 = ProbabilisticBit(1)
        p03 = ProbabilisticBit(0.3)
        p07 = ProbabilisticBit(0.7)
        
        # Операция AND
        self.assertEqual((p0 & p0).probability, 0)
        self.assertEqual((p0 & p1).probability, 0)
        self.assertEqual((p1 & p1).probability, 1)
        self.assertAlmostEqual((p03 & p07).probability, 0.3 * 0.7)
        
        # Операция OR
        self.assertEqual((p0 | p0).probability, 0)
        self.assertEqual((p0 | p1).probability, 1)
        self.assertEqual((p1 | p1).probability, 1)
        self.assertAlmostEqual((p03 | p07).probability, 0.3 + 0.7 - 0.3 * 0.7)
        
        # Операция NOT
        self.assertEqual((~p0).probability, 1)
        self.assertEqual((~p1).probability, 0)
        self.assertAlmostEqual((~p03).probability, 0.7)
    
    def test_equality(self):
        """Тест сравнения вероятностных битов."""
        p0 = ProbabilisticBit(0)
        p1 = ProbabilisticBit(1)
        p05 = ProbabilisticBit(0.5)
        
        self.assertEqual(p0, ProbabilisticBit(0))
        self.assertEqual(p1, ProbabilisticBit(1))
        self.assertEqual(p05, ProbabilisticBit(0.5))
        self.assertNotEqual(p0, p1)
        
        # Сравнение с числами
        self.assertEqual(p0, 0)
        self.assertEqual(p1, 1)
        self.assertEqual(p05, 0.5)
    
    def test_sampling(self):
        """Тест генерации случайных значений."""
        # Фиксируем seed для воспроизводимости
        rng = np.random.RandomState(42)
        
        p0 = ProbabilisticBit(0)
        p1 = ProbabilisticBit(1)
        
        # Для крайних значений результат детерминирован
        self.assertEqual(p0.sample(rng), 0)
        self.assertEqual(p1.sample(rng), 1)
        
        # Для промежуточных значений проверяем статистику
        p05 = ProbabilisticBit(0.5)
        samples = [p05.sample(rng) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        
        # Проверяем, что среднее близко к 0.5
        self.assertAlmostEqual(mean, 0.5, delta=0.05)


class TestPBit(unittest.TestCase):
    """Тесты для класса PBit."""
    
    def test_creation(self):
        """Тест создания P-бита."""
        pb_pos = PBit(0.3, +1)
        pb_neg = PBit(0.7, -1)
        
        self.assertEqual(pb_pos.probability, 0.3)
        self.assertEqual(pb_pos.sign, +1)
        self.assertEqual(pb_neg.probability, 0.7)
        self.assertEqual(pb_neg.sign, -1)
        
        # Проверка некорректных значений
        with self.assertRaises(ValueError):
            PBit(-0.1, +1)
        with self.assertRaises(ValueError):
            PBit(0.5, 0)  # Знак должен быть +1 или -1
    
    def test_operations(self):
        """Тест операций над P-битами."""
        pb1 = PBit(0.3, +1)
        pb2 = PBit(0.7, -1)
        
        # Операция AND
        result = pb1 & pb2
        self.assertAlmostEqual(result.probability, 0.3 * 0.7)
        self.assertEqual(result.sign, -1)  # +1 * -1 = -1
        
        # Операция OR с изменением знака
        result = pb1 | pb2
        p_expected = 0.3 + 0.7 - 0.3 * 0.7
        self.assertAlmostEqual(result.probability, p_expected)
        # Знак определяется произведением знаков и знаком (p1 + p2 - 1)
        # p1 + p2 - 1 = 0.3 + 0.7 - 1 = 0 >= 0, поэтому знак = +1 * -1 * +1 = -1
        self.assertEqual(result.sign, -1)
        
        # Операция NOT
        result = ~pb1
        self.assertAlmostEqual(result.probability, 1 - 0.3)
        self.assertEqual(result.sign, -1)
    
    def test_equality(self):
        """Тест сравнения P-битов."""
        pb1 = PBit(0.3, +1)
        pb2 = PBit(0.3, -1)
        
        self.assertEqual(pb1, PBit(0.3, +1))
        self.assertNotEqual(pb1, pb2)  # Разные знаки
        self.assertNotEqual(pb1, PBit(0.4, +1))  # Разные вероятности


class TestPhaseState(unittest.TestCase):
    """Тесты для класса PhaseState."""
    
    def test_creation(self):
        """Тест создания фазового состояния."""
        ps1 = PhaseState(0.5, 0)
        ps2 = PhaseState(0.7, np.pi)
        
        self.assertEqual(ps1.probability, 0.5)
        self.assertEqual(ps1.phase, 0)
        self.assertEqual(ps2.probability, 0.7)
        self.assertEqual(ps2.phase, np.pi)
        
        # Проверка нормализации фазы
        ps3 = PhaseState(0.5, 3 * np.pi)
        self.assertAlmostEqual(ps3.phase, np.pi)
    
    def test_operations(self):
        """Тест операций над фазовыми состояниями."""
        ps1 = PhaseState(0.5, 0)
        ps2 = PhaseState(0.5, np.pi)
        
        # Операция AND
        result = ps1 & ps2
        self.assertAlmostEqual(result.probability, 0.5 * 0.5)
        self.assertAlmostEqual(result.phase, 0 + np.pi)
        
        # Операция OR - проверка деструктивной интерференции
        result = ps1 | ps2
        # При противоположных фазах с одинаковой вероятностью ожидаем
        # полное гашение (p близко к 0)
        self.assertAlmostEqual(result.probability, 0, delta=1e-10)
        
        # Операция NOT
        result = ~ps1
        self.assertAlmostEqual(result.probability, 1 - 0.5)
        self.assertAlmostEqual(result.phase, np.pi)  # Фаза сдвигается на π
    
    def test_complex_conversion(self):
        """Тест преобразования между фазовым состоянием и комплексным числом."""
        ps = PhaseState(0.5, np.pi/4)
        z = ps.to_complex()
        
        # Восстанавливаем фазовое состояние из комплексного числа
        ps_restored = PhaseState.from_complex(z)
        
        self.assertAlmostEqual(ps.probability, ps_restored.probability)
        self.assertAlmostEqual(ps.phase, ps_restored.phase)


class TestQuantumState(unittest.TestCase):
    """Тесты для класса QuantumState."""
    
    def test_creation(self):
        """Тест создания квантового состояния."""
        # Базисные состояния
        q0 = QuantumState([1, 0])
        q1 = QuantumState([0, 1])
        
        # Проверка амплитуд
        self.assertTrue(np.allclose(q0.amplitudes, np.array([1, 0])))
        self.assertTrue(np.allclose(q1.amplitudes, np.array([0, 1])))
        
        # Проверка нормализации
        q = QuantumState([2, 0])
        self.assertTrue(np.allclose(q.amplitudes, np.array([1, 0])))
        
        # Проверка создания из комплексных амплитуд
        q = QuantumState([1/np.sqrt(2), 1j/np.sqrt(2)])
        self.assertTrue(np.allclose(np.abs(q.amplitudes)**2, np.array([0.5, 0.5])))
        
        # Проверка некорректной размерности
        with self.assertRaises(ValueError):
            QuantumState([1, 0, 0])  # Только кубиты поддерживаются
    
    def test_operations(self):
        """Тест операций над квантовыми состояниями."""
        # Базисные состояния
        q0 = QuantumState([1, 0])  # |0⟩
        q1 = QuantumState([0, 1])  # |1⟩
        
        # Операция AND (моделирует эффект проецирования на |11⟩)
        result = q0 & q1
        # Ожидаем амплитуду близкую к 0 для состояния |1⟩
        self.assertAlmostEqual(np.abs(result.amplitudes[1])**2, 0, delta=1e-10)
        
        # Операция OR (создает равную суперпозицию)
        result = q0 | q1
        # Ожидаем равные вероятности для |0⟩ и |1⟩
        self.assertAlmostEqual(np.abs(result.amplitudes[0])**2, 0.5, delta=1e-10)
        self.assertAlmostEqual(np.abs(result.amplitudes[1])**2, 0.5, delta=1e-10)
        
        # Операция NOT (применяет оператор Паули-X)
        result = ~q0
        self.assertTrue(np.allclose(result.amplitudes, np.array([0, 1])))
        
        # И обратно
        result = ~q1
        self.assertTrue(np.allclose(result.amplitudes, np.array([1, 0])))
    
    def test_measurement(self):
        """Тест измерения квантового состояния."""
        # Фиксируем seed для воспроизводимости
        rng = np.random.RandomState(42)
        
        # Базисные состояния
        q0 = QuantumState([1, 0])  # |0⟩
        q1 = QuantumState([0, 1])  # |1⟩
        
        # Измерение базисных состояний детерминировано
        self.assertEqual(q0.measure(rng), 0)
        self.assertEqual(q1.measure(rng), 1)
        
        # Равная суперпозиция
        q_plus = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Проверяем статистику измерений
        measurements = [q_plus.measure(rng) for _ in range(1000)]
        mean = sum(measurements) / len(measurements)
        
        # Ожидаем среднее близкое к 0.5
        self.assertAlmostEqual(mean, 0.5, delta=0.05)
    
    def test_bloch_sphere(self):
        """Тест преобразования между квантовым состоянием и координатами на сфере Блоха."""
        # Состояния на оси Z
        q0 = QuantumState([1, 0])  # |0⟩, северный полюс
        q1 = QuantumState([0, 1])  # |1⟩, южный полюс
        
        theta0, phi0 = q0.to_bloch()
        theta1, phi1 = q1.to_bloch()
        
        self.assertAlmostEqual(theta0, 0)  # Северный полюс
        self.assertAlmostEqual(theta1, np.pi)  # Южный полюс
        
        # Состояния на экваторе
        q_plus = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩
        q_minus = QuantumState([1/np.sqrt(2), -1/np.sqrt(2)])  # |-⟩
        
        theta_plus, phi_plus = q_plus.to_bloch()
        theta_minus, phi_minus = q_minus.to_bloch()
        
        self.assertAlmostEqual(theta_plus, np.pi/2)  # Экватор
        self.assertAlmostEqual(theta_minus, np.pi/2)  # Экватор
        
        self.assertAlmostEqual(phi_plus, 0)  # Положительная ось X
        self.assertAlmostEqual(phi_minus, np.pi)  # Отрицательная ось X
        
        # Обратное преобразование
        q_restored = QuantumState.from_bloch(theta_plus, phi_plus)
        self.assertTrue(np.allclose(np.abs(q_restored.amplitudes), np.abs(q_plus.amplitudes)))


class TestOperators(unittest.TestCase):
    """Тесты для операторов перехода между уровнями."""
    
    def test_lift_operators(self):
        """Тест операторов подъема."""
        # Классический бит 1
        c_bit = ClassicalBit(1)
        
        # Подъем до вероятностного бита
        p_bit = lift(c_bit, 2)
        self.assertIsInstance(p_bit, ProbabilisticBit)
        self.assertEqual(p_bit.probability, 1)
        
        # Подъем до P-бита
        pb_bit = lift(p_bit, 3)
        self.assertIsInstance(pb_bit, PBit)
        self.assertEqual(pb_bit.probability, 1)
        self.assertEqual(pb_bit.sign, +1)
        
        # Подъем до фазового состояния
        ph_state = lift(pb_bit, 4)
        self.assertIsInstance(ph_state, PhaseState)
        self.assertEqual(ph_state.probability, 1)
        self.assertEqual(ph_state.phase, 0)
        
        # Подъем до квантового состояния
        q_state = lift(ph_state, 5)
        self.assertIsInstance(q_state, QuantumState)
        self.assertTrue(np.allclose(q_state.amplitudes, np.array([1, 0])))
        
        # Проверка композиции операторов (прямой подъем с 1 на 3)
        pb_bit_direct = lift(c_bit, 3)
        self.assertIsInstance(pb_bit_direct, PBit)
        self.assertEqual(pb_bit_direct.probability, pb_bit.probability)
        self.assertEqual(pb_bit_direct.sign, pb_bit.sign)
    
    def test_project_operators(self):
        """Тест операторов проекции."""
        # Квантовое состояние |+⟩
        q_state = QuantumState([1/np.sqrt(2), 1/np.sqrt(2)])
        
        # Проекция на фазовое состояние
        ph_state = project(q_state, 4)
        self.assertIsInstance(ph_state, PhaseState)
        self.assertAlmostEqual(ph_state.probability, 0.5)
        self.assertAlmostEqual(ph_state.phase, 0)
        
        # Проекция на P-бит
        pb_bit = project(ph_state, 3)
        self.assertIsInstance(pb_bit, PBit)
        self.assertAlmostEqual(pb_bit.probability, 0.5)
        self.assertEqual(pb_bit.sign, +1)
        
        # Проекция на вероятностный бит
        p_bit = project(pb_bit, 2)
        self.assertIsInstance(p_bit, ProbabilisticBit)
        self.assertAlmostEqual(p_bit.probability, 0.5)
        
        # Проекция на классический бит
        c_bit = project(p_bit, 1)
        self.assertIsInstance(c_bit, ClassicalBit)
        self.assertEqual(c_bit.value, 0 if p_bit.probability < 0.5 else 1)
        
        # Проверка композиции операторов (прямая проекция с 5 на 2)
        p_bit_direct = project(q_state, 2)
        self.assertIsInstance(p_bit_direct, ProbabilisticBit)
        self.assertAlmostEqual(p_bit_direct.probability, p_bit.probability)


if __name__ == '__main__':
    unittest.main()
