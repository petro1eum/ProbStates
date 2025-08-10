import unittest
import numpy as np

from probstates import PhaseRegister, deutsch_jozsa


class TestPhaseRegister(unittest.TestCase):
    def test_uniform_and_hadamard_inverse(self):
        # Проверка, что H^{⊗n}^2 = I (в пределах численной точности)
        n = 3
        reg = PhaseRegister.uniform(n)
        # Сделаем произвольное состояние и проверим инволютивность
        rng = np.random.default_rng(123)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=1 << n)
        probs = rng.random(size=1 << n)
        probs /= probs.sum()
        reg = PhaseRegister.from_prob_and_phase(probs, phases)
        v0 = reg.amplitudes
        reg.hadamard_all()
        reg.hadamard_all()
        v2 = reg.amplitudes
        self.assertTrue(np.allclose(v0, v2, rtol=1e-10, atol=1e-10))

    def test_oracle_phase_flip(self):
        # Для константного оракула f(x)=1 на каждом x должны флипнуться фазы (умножение на -1)
        n = 2
        reg = PhaseRegister.uniform(n)
        v0 = reg.amplitudes

        def f_one(x: int) -> int:
            return 1

        reg.apply_oracle(f_one)
        v1 = reg.amplitudes
        self.assertTrue(np.allclose(v1, -v0, rtol=1e-12, atol=1e-12))

    def test_deutsch_jozsa_constant_and_balanced(self):
        n = 3

        def f_const0(x: int) -> int:
            return 0

        def f_const1(x: int) -> int:
            return 1

        def f_parity(x: int) -> int:
            return bin(x).count("1") & 1

        def f_msb(x: int) -> int:
            return (x >> (n - 1)) & 1

        kind0, p0_0 = deutsch_jozsa(f_const0, n)
        kind1, p0_1 = deutsch_jozsa(f_const1, n)
        kindp, p0_p = deutsch_jozsa(f_parity, n)
        kindm, p0_m = deutsch_jozsa(f_msb, n)

        self.assertEqual(kind0, "constant")
        self.assertTrue(np.isclose(p0_0, 1.0, rtol=1e-12, atol=1e-12))

        self.assertEqual(kind1, "constant")
        self.assertTrue(np.isclose(p0_1, 1.0, rtol=1e-12, atol=1e-12))

        self.assertEqual(kindp, "balanced")
        self.assertLess(p0_p, 1e-10)

        self.assertEqual(kindm, "balanced")
        self.assertLess(p0_m, 1e-10)


if __name__ == "__main__":
    unittest.main()


