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

__version__ = '0.1.0'
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

# Entropy module is available but not imported by default
# to avoid unnecessary dependencies
