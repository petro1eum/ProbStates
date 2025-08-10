import unittest
import numpy as np

from probstates import (
    ProbabilisticBit,
    PBit,
    PhaseState,
)
from probstates.entropy import (
    shannon_entropy,
    kl_divergence,
    entropy_level3,
    entropy_level4,
    information_loss,
)
from probstates.operators import project, lift


def _angle_close(a: float, b: float, atol: float = 1e-9) -> bool:
    # Сравнение углов по модулю 2π
    da = (a - b) % (2 * np.pi)
    if da > np.pi:
        da = 2 * np.pi - da
    return abs(da) <= atol


class TestTheorems(unittest.TestCase):
    def test_oplus4_commutativity_and_identity(self):
        rng = np.random.default_rng(2025)
        for _ in range(200):
            p1, p2 = rng.random(), rng.random()
            phi1, phi2 = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
            a = PhaseState(p1, phi1)
            b = PhaseState(p2, phi2)
            c1 = a | b
            c2 = b | a
            self.assertTrue(np.isclose(c1.probability, c2.probability, rtol=1e-12, atol=1e-12))
            self.assertTrue(_angle_close(c1.phase, c2.phase, atol=1e-9))

        # Нейтральный элемент: (0, e^{iφ})
        x = PhaseState(0.0, 0.0)
        y = PhaseState(0.37, 1.23)
        z = x | y
        self.assertTrue(np.isclose(z.probability, y.probability, rtol=1e-12, atol=1e-12))
        self.assertTrue(_angle_close(z.phase, y.phase, atol=1e-12))

    def test_oplus4_probability_bounds(self):
        # Вероятность результата всегда в [0,1] (из-за клиппинга по определению состояния)
        rng = np.random.default_rng(7)
        for _ in range(200):
            a = PhaseState(rng.random(), rng.uniform(0, 2*np.pi))
            b = PhaseState(rng.random(), rng.uniform(0, 2*np.pi))
            c = a | b
            self.assertGreaterEqual(c.probability, 0.0)
            self.assertLessEqual(c.probability, 1.0)

    def test_projection_P4_to_3(self):
        # P_{4→3}(p,φ) = (p, sign(cos φ))
        for phi in np.linspace(0, 2*np.pi, 37, endpoint=False):
            p = 0.42
            s_expected = 1 if np.cos(phi) >= 0 else -1
            st4 = PhaseState(p, phi)
            st3 = project(st4, 3)
            self.assertEqual(st3.sign, s_expected)
            self.assertTrue(np.isclose(st3.probability, p))

    def test_entropy_level3_matches_definition(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            p = rng.uniform(1e-6, 1-1e-6)
            # s = +1 → H3 = H(p)
            h3_pos = entropy_level3(PBit(p, +1))
            self.assertTrue(np.isclose(h3_pos, shannon_entropy(p), rtol=1e-10, atol=1e-10))
            # s = -1 → H3 = H(p) + KL(p || 1-p)
            h3_neg = entropy_level3(PBit(p, -1))
            self.assertTrue(np.isclose(h3_neg, shannon_entropy(p) + kl_divergence(p, 1-p), rtol=1e-10, atol=1e-10))

    def test_entropy_level4_phase_independence(self):
        rng = np.random.default_rng(13)
        for _ in range(40):
            p = rng.uniform(0.0, 1.0)
            values = [entropy_level4(PhaseState(p, phi)) for phi in [0.0, 0.5, 1.3, 2.0, 3.1]]
            # Все значения почти равны
            self.assertTrue(np.allclose(values, values[0], rtol=1e-9, atol=1e-9))

    def test_lift_then_project_identities(self):
        # P_{2->1}(L_{1->2}(b)) = b
        b0 = ProbabilisticBit(0.0)  # via lifting from ClassicalBit(0)
        self.assertEqual(project(lift(ProbabilisticBit(0.0), 3), 2).probability, 0.0)
        # P_{3->2}(L_{2->3}(p)) = p
        rng = np.random.default_rng(5)
        for _ in range(20):
            p = rng.uniform(0.0, 1.0)
            self.assertTrue(np.isclose(project(lift(ProbabilisticBit(p), 3), 2).probability, p))
        # P_{4->3}(L_{3->4}(p,s)) = (p, s) up to sign(cos φ_s) where φ_s∈{0,π}
        for s in (+1, -1):
            st3 = PBit(0.42, s)
            back = project(lift(st3, 4), 3)
            self.assertEqual(back.sign, s)
            self.assertTrue(np.isclose(back.probability, st3.probability))

    def test_projection_compositions(self):
        # P_{5->2} == P_{3->2} ∘ P_{4->3} ∘ P_{5->4}
        rng = np.random.default_rng(23)
        for _ in range(20):
            # Генерируем произвольное квантовое состояние (один кубит)
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2*np.pi)
            # |ψ> = cos(theta/2)|0> + e^{iφ} sin(theta/2)|1>
            alpha = np.cos(theta/2)
            beta = np.exp(1j * phi) * np.sin(theta/2)
            from probstates import QuantumState
            q = QuantumState([alpha, beta])
            p52 = project(q, 2).probability
            p54 = project(q, 4)
            p43 = project(p54, 3)
            p32 = project(p43, 2).probability
            self.assertTrue(np.isclose(p52, p32, rtol=1e-12, atol=1e-12))

    def test_information_loss_phase_to_pbit_nonmonotonic(self):
        # Теорема 7.3.1: классическая (измеряемая) энтропия может увеличиваться после проекции.
        # Следовательно, information_loss(уровень4 → уровень3) может быть как >0, так и <0.
        rng = np.random.default_rng(29)
        losses = []
        for _ in range(100):
            st4 = PhaseState(rng.uniform(0, 1), rng.uniform(0, 2*np.pi))
            st3 = project(st4, 3)
            losses.append(information_loss(st4, st3))
        self.assertLess(min(losses), -1e-6)
        self.assertGreater(max(losses), 1e-6)

    def test_shannon_entropy_bounds(self):
        # 0 ≤ H(p) ≤ 1, H(0)=H(1)=0, H(0.5)=1
        self.assertTrue(np.isclose(shannon_entropy(0.0), 0.0))
        self.assertTrue(np.isclose(shannon_entropy(1.0), 0.0))
        self.assertTrue(np.isclose(shannon_entropy(0.5), 1.0))
        rng = np.random.default_rng(31)
        for _ in range(50):
            p = rng.uniform(0.0, 1.0)
            h = shannon_entropy(p)
            self.assertGreaterEqual(h, 0.0)
            self.assertLessEqual(h, 1.0)

    def test_kl_divergence_properties(self):
        rng = np.random.default_rng(37)
        for _ in range(50):
            p = rng.uniform(1e-6, 1-1e-6)
            q = rng.uniform(1e-6, 1-1e-6)
            d = kl_divergence(p, q)
            self.assertGreaterEqual(d, 0.0)
            if np.isclose(p, q):
                self.assertTrue(np.isclose(d, 0.0))

    def test_deutsch_jozsa_small_n(self):
        from probstates import deutsch_jozsa
        # n=2: проверим константную и две сбалансированных
        n = 2
        f_const0 = lambda x: 0
        f_const1 = lambda x: 1
        # balanced: половина единиц
        def f_half_low(x: int) -> int:
            return 1 if x in (0, 1) else 0
        def f_half_high(x: int) -> int:
            return 1 if x in (2, 3) else 0
        for f, expected in [
            (f_const0, 'constant'),
            (f_const1, 'constant'),
            (f_half_low, 'balanced'),
            (f_half_high, 'balanced'),
        ]:
            kind, p0 = deutsch_jozsa(f, n)
            self.assertEqual(kind, expected)


if __name__ == "__main__":
    unittest.main()


