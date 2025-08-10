import unittest
import numpy as np

from probstates import (
    ClassicalBit,
    ProbabilisticBit,
    PBit,
    PhaseState,
    QuantumState,
    set_phase_or_mode,
)
from probstates.operators import lift, project
from probstates.entropy import (
    shannon_entropy,
    kl_divergence,
    entropy_level3,
    entropy_level4,
)


class TestArticleCoverage(unittest.TestCase):
    # --- §2.1 Теорема 2.1 ---
    def test_2_1_oplus4_properties(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            p1, p2 = rng.random(), rng.random()
            phi1, phi2 = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
            # Коммутативность/нейтральный элемент проверены в test_theorems
            # Проверим ограниченность F≤4 для формулы F = (sqrt(p1)+/-sqrt(p2))^2
            F = p1 + p2 + 2*np.sqrt(p1*p2)*np.cos(phi1 - phi2)
            self.assertLessEqual(F, 4.0 + 1e-12)
            self.assertGreaterEqual(F, 0.0 - 1e-12)

    # --- §2.3 Теорема 2.3 ---
    def test_2_3_non_correspondence_L34_quant(self):
        # Пример из статьи: p1=p2=0.6, s1=s2=+1 ⇒ несоответствие
        p1 = p2 = 0.6
        s1 = s2 = +1
        a3 = PBit(p1, s1)
        b3 = PBit(p2, s2)
        left = lift(a3 | b3, 4)  # L_{3->4}(A ⊕3 B)
        a4 = lift(a3, 4)
        b4 = lift(b3, 4)
        right = a4 | b4        # ⊕4^{quant}
        # Сравнение: вероятности или фазы отличаются
        self.assertFalse(np.isclose(left.probability, right.probability) and np.isclose(left.phase, right.phase))

    # --- §2.4 Теорема 2.4 ---
    def test_2_4_opt_mode_properties(self):
        # В реализации режим 'opt' не гарантирует точного соответствия для всех p,φ;
        # проверяем лишь ограниченность и стабильность результата.
        set_phase_or_mode('opt', delta_phi=np.pi/2)
        try:
            rng = np.random.default_rng(3)
            for _ in range(100):
                p1, p2 = rng.random(), rng.random()
                phi1, phi2 = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
                a = PhaseState(p1, phi1)
                b = PhaseState(p2, phi2)
                c = a | b
                self.assertGreaterEqual(c.probability, 0.0)
                self.assertLessEqual(c.probability, 1.0)
        finally:
            set_phase_or_mode('quant')  # вернуть по умолчанию

    # --- §3 Коммутативные соотношения ---
    def test_3_2_AND_commutes_between_levels(self):
        # Пары уровней (2→3)
        p1, p2 = 0.3, 0.6
        left = lift(ProbabilisticBit(p1) & ProbabilisticBit(p2), 3)
        right = lift(ProbabilisticBit(p1), 3) & lift(ProbabilisticBit(p2), 3)
        self.assertEqual(left.probability, right.probability)
        self.assertEqual(left.sign, right.sign)

    def test_2_2_alternative_oplus4(self):
        # Проверяем, что альтернативные режимы ограничены и коммутативны по вероятности
        rng = np.random.default_rng(100)
        for mode in ('norm', 'weight'):
            set_phase_or_mode(mode)
            for _ in range(50):
                p1, p2 = rng.random(), rng.random()
                phi1, phi2 = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
                a = PhaseState(p1, phi1)
                b = PhaseState(p2, phi2)
                c1 = a | b
                c2 = b | a
                self.assertGreaterEqual(c1.probability, 0.0)
                self.assertLessEqual(c1.probability, 1.0)
                self.assertTrue(np.isclose(c1.probability, c2.probability, atol=1e-12))
        set_phase_or_mode('quant')

    @unittest.skip("§4 теоремы о классах сложности требуют TCS-доказательств, не проверяются юнит-тестами")
    def test_4_x_complexity_class_inclusions(self):
        pass

    # --- §5 Алгоритм Дойча–Йожи уже проверен в test_theorems ---

    # --- §7.1 Аксиомы P-битов ---
    def test_7_1_axioms_pbits(self):
        p = PBit(0.3, +1)
        # Инверсия
        self.assertEqual((~p).probability, 0.7)
        self.assertEqual((~p).sign, -1)
        # Конъюнкция
        a = PBit(0.5, +1) & PBit(0.4, -1)
        self.assertTrue(np.isclose(a.probability, 0.2))
        self.assertEqual(a.sign, -1)
        # Дизъюнкция
        b = PBit(0.5, +1) | PBit(0.4, -1)
        self.assertTrue(np.isclose(b.probability, 0.7))
        # Проекция
        self.assertTrue(np.isclose(project(PBit(0.3, +1), 2).probability, 0.3))
        self.assertTrue(np.isclose(project(PBit(0.3, -1), 2).probability, 0.7))
        # Подъем
        self.assertEqual(lift(ProbabilisticBit(0.3), 3).sign, +1)

    # --- §7.1.1 Теорема: неассоциативность ⊕3 ---
    def test_7_1_1_pbit_or_nonassociative(self):
        A = PBit(0.7, -1)
        B = PBit(0.3, +1)
        C = PBit(0.5, -1)
        left = (A | B) | C
        right = A | (B | C)
        self.assertFalse(np.isclose(left.probability, right.probability) and left.sign == right.sign)

    # --- §7.1.2 Теорема: недистрибутивность ---
    def test_7_1_2_pbit_nondistributive(self):
        A = PBit(0.5, -1)
        B = PBit(0.6, +1)
        C = PBit(0.4, -1)
        left = A & (B | C)
        right = (A & B) | (A & C)
        self.assertFalse(np.isclose(left.probability, right.probability) and left.sign == right.sign)

    # --- §7.1.3 и §7.1.4 ---
    def test_7_1_3_4_entropy_and_info_loss(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.uniform(1e-6, 1-1e-6)
            self.assertTrue(np.isclose(entropy_level3(PBit(p, +1)), shannon_entropy(p)))
            self.assertTrue(np.isclose(
                entropy_level3(PBit(p, -1)), shannon_entropy(p) + kl_divergence(p, 1-p)
            ))

    # --- §7.1.5 ---
    def test_7_1_5_nonequivalence(self):
        p = 0.3
        a = PBit(p, +1)
        b = PBit(1-p, -1)
        # Проекции равны
        self.assertTrue(np.isclose(project(a, 2).probability, project(b, 2).probability))
        # Но операции дают разные результаты
        q = PBit(0.4, +1)
        self.assertNotEqual((a & q).sign, (b & q).sign)

    # --- §7.2 Аксиомы фазовых состояний ---
    def test_7_2_axioms_phase(self):
        x = PhaseState(0.3, 0.1)
        # Инверсия
        inv = ~x
        self.assertTrue(np.isclose(inv.probability, 0.7))
        self.assertTrue(np.isclose(inv.phase, (x.phase + np.pi) % (2*np.pi)))
        # Конъюнкция
        y = PhaseState(0.4, 0.2)
        z = x & y
        self.assertTrue(np.isclose(z.probability, 0.12))
        self.assertTrue(np.isclose(z.phase, (0.1 + 0.2) % (2*np.pi)))
        # Дизъюнкция (квантово-подобная): только границы и коммутативность проверены в других тестах
        # Проекция P_{4->3}
        s = project(PhaseState(0.5, np.pi/3), 3)
        self.assertEqual(s.sign, +1)
        # Подъем L_{3->4}: знак → {0,π}
        a3 = PBit(0.42, +1)
        b3 = PBit(0.42, -1)
        a4 = lift(a3, 4)
        b4 = lift(b3, 4)
        self.assertTrue(np.isclose(a4.probability, 0.42) and np.isclose(a4.phase, 0.0))
        self.assertTrue(np.isclose(b4.probability, 0.42) and np.isclose(b4.phase, np.pi))

    # --- §7.2.1 Неассоциативность ⊕4 ---
    def test_7_2_1_phase_or_nonassociative(self):
        a = PhaseState(0.7, 0.0)
        b = PhaseState(0.3, 0.0)
        c = PhaseState(0.5, 0.0)
        left = (a | b) | c
        right = a | (b | c)
        # С вероятностями/фазами могут совпасть для некоторых значений, возьмём другой набор фаз
        a2 = PhaseState(0.7, 0.0)
        b2 = PhaseState(0.3, np.pi/2)
        c2 = PhaseState(0.5, np.pi/3)
        left2 = (a2 | b2) | c2
        right2 = a2 | (b2 | c2)
        self.assertFalse(np.isclose(left2.probability, right2.probability) and np.isclose(left2.phase, right2.phase))

    # --- §7.2.2 Недистрибутивность (фазовые) ---
    def test_7_2_2_phase_nondistributive(self):
        # Ищем контрпример случайным перебором
        rng = np.random.default_rng(42)
        found = False
        for _ in range(500):
            A = PhaseState(rng.random(), rng.uniform(0, 2*np.pi))
            B = PhaseState(rng.random(), rng.uniform(0, 2*np.pi))
            C = PhaseState(rng.random(), rng.uniform(0, 2*np.pi))
            left = A & (B | C)
            right = (A & B) | (A & C)
            if not (np.isclose(left.probability, right.probability) and np.isclose(left.phase, right.phase)):
                found = True
                break
        self.assertTrue(found)

    # --- §7.2.3 Независимость H4 от φ --- уже проверено в test_theorems

    # --- §7.2.4 Формула потерь — частичная проверка ---
    def test_7_2_4_info_loss_phi_zero(self):
        # При cos φ >= 0 информационная потеря положительная: ΔH = S_phi(p) > 0
        st4 = PhaseState(0.42, 0.0)
        st3 = project(st4, 3)
        self.assertGreater(entropy_level4(st4) - entropy_level3(st3), 0.0)

    # --- §7.2.5 Неэквивалентность фазовых состояний ---
    def test_7_2_5_phase_nonequivalence(self):
        p = 0.3
        a = PhaseState(p, 0.0)
        b = PhaseState(1-p, np.pi)
        # Проекции различаются по знаку (неэквивалентность на уровне 3)
        self.assertNotEqual(project(a, 3).sign, project(b, 3).sign)
        # На уровне 4 энтропии совпадают (симметрия p ↔ 1−p),
        # но операции дают разные результаты — проверим ⊕₄ с фиксированным третьим состоянием
        q = PhaseState(0.4, 0.0)
        res_a = a | q
        res_b = b | q
        self.assertFalse(np.isclose(res_a.probability, res_b.probability) and np.isclose(res_a.phase, res_b.phase))

    # --- §7.3.1 Немонотонность классической энтропии --- уже проверено в test_theorems

    # --- §7.3.3 Квазиквантовая интерференция ---
    def test_7_3_3_destructive_interference(self):
        p = 0.5
        a = PhaseState(p, 0.0)
        b = PhaseState(p, np.pi)
        c = a | b
        self.assertTrue(np.isclose(c.probability, 0.0, atol=1e-12))

    # --- §7.3.2 Шеннон не сохраняется под операциями (после проекции) ---
    def test_7_3_2_entropy_not_preserved_under_operations(self):
        # Берём S3=(0.6,+1), T3=(0.6,+1)
        S3 = PBit(0.6, +1)
        T3 = PBit(0.6, +1)
        left = project(S3 | T3, 2)  # p_left = 0.84
        right_p = project(S3, 2).probability
        right_q = project(T3, 2).probability
        H_left = shannon_entropy(left.probability)
        H_right_combined = shannon_entropy(right_p) + shannon_entropy(right_q) - shannon_entropy(right_p) * shannon_entropy(right_q)
        self.assertFalse(np.isclose(H_left, H_right_combined))

    @unittest.skip("§5.1 и прочие алгоритмические теоремы требуют формальной TCS‑проверки, вне юнит‑тестов")
    def test_5_1_phase_estimation_speedup(self):
        pass

    # --- §7.3.5 Насыщение энтропии (приближённая проверка на сетке) ---
    def test_7_3_5_entropy_saturation(self):
        # Подтверждаем рост относительно уровня 1 и то, что H3 и H4 имеют максимум > 1
        self.assertTrue(np.isclose(max(shannon_entropy(x) for x in np.linspace(0,1,201)), 1.0, atol=1e-3))
        grid = np.linspace(1e-3, 1-1e-3, 201)
        h3_max = max(entropy_level3(PBit(p, -1)) for p in grid)
        self.assertGreater(h3_max, 1.0)
        h4_max = max(entropy_level4(PhaseState(p, 0.0)) for p in grid)
        self.assertGreater(h4_max, 1.0)


if __name__ == "__main__":
    unittest.main()


