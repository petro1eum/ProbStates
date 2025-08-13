# probstates/__init__.py
"""
ProbStates: Библиотека для работы с иерархией вероятностных состояний

Эта библиотека реализует пятиуровневый формализм вероятностных состояний:
1. Классические биты
2. Вероятностные биты
3. P-биты (вероятностные биты с полярностью)
4. Фазовые состояния
5. Квантовые состояния

Каждый уровень предоставляет свои реализации операций AND, OR и NOT,
а также включает функционал для перехода между уровнями.
"""

from probstates.classical import ClassicalBit
from probstates.probabilistic import ProbabilisticBit
from probstates.pbit import PBit
from probstates.phase import PhaseState, set_phase_or_mode, get_phase_or_mode, set_phase_or_custom
from probstates.quantum import QuantumState
from probstates.operators import lift, project
from probstates.phase_register import PhaseRegister, deutsch_jozsa
from probstates.coherence import coherence_l1, dephase, phase_drift, amp_damp, coherence_under_noise
from probstates.correspondence import correspondence_error
from probstates.markets import (
    sma, momentum, rsi,
    indicator_to_prob, sentiment_to_phase,
    FeatureSpec, make_phase_states, aggregate_specs,
    btc_signal_from_arrays,
)
from probstates.calculus import (
    central_diff,
    l2_or, grad_l2_or, l2_and, grad_l2_and,
    shannon_entropy as shannon_entropy_calculus,
    d_shannon_entropy_dp,
    l4_quant_or, grad_l4_quant_or, phase_sensitivity,
    expected_l2_or_under_beta,
    mc_expected_l4_quant_or_under_beta,
    kappa_from_phases, rho_from_ps,
)
# Optional: transformer requires PyTorch; import lazily if available
try:
    from probstates.transformer import ProbStatesTransformer, train_transformer
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

__version__ = '0.2.0'
__all__ = [
    'ClassicalBit', 
    'ProbabilisticBit', 
    'PBit', 
    'PhaseState', 
    'QuantumState',
    'lift',
    'project',
    'PhaseRegister',
    'deutsch_jozsa',
    'set_phase_or_mode',
    'get_phase_or_mode',
    'set_phase_or_custom',
    'coherence_l1', 'dephase', 'phase_drift', 'amp_damp', 'coherence_under_noise',
    'correspondence_error'
]

if _HAS_TORCH:
    __all__ += ['ProbStatesTransformer', 'train_transformer']

# Entropy module is available but not imported by default
# to avoid unnecessary dependencies
