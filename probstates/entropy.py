"""
probstates/entropy.py

This module implements entropy characteristics for different levels of the 
probabilistic state hierarchy, from classical Shannon entropy to quasi-quantum entropy.
"""

import numpy as np
from typing import Optional, Union, Tuple, List

from probstates.base import State
from probstates.classical import ClassicalBit
from probstates.probabilistic import ProbabilisticBit
from probstates.pbit import PBit
from probstates.phase import PhaseState
from probstates.quantum import QuantumState
from probstates.operators import project


def shannon_entropy(p: float) -> float:
    """
    Calculate the Shannon entropy for a binary probability distribution.
    
    H(p) = -p*log_2(p) - (1-p)*log_2(1-p)
    
    Args:
        p: Probability value (0 ≤ p ≤ 1)
        
    Returns:
        Shannon entropy value between 0 and 1
    """
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)


def kl_divergence(p: float, q: float) -> float:
    """
    Calculate the Kullback-Leibler divergence between two Bernoulli distributions.
    
    D_KL(B_p || B_q) = p*log_2(p/q) + (1-p)*log_2((1-p)/(1-q))
    
    Args:
        p: Parameter of the first Bernoulli distribution
        q: Parameter of the second Bernoulli distribution
        
    Returns:
        KL divergence (may be infinity when p=0,q>0 or p=1,q<1)
    """
    if p == 0:
        return 0 if q == 0 else np.inf
    if p == 1:
        return 0 if q == 1 else np.inf
    if q == 0 or q == 1:
        return np.inf if q != p else 0
    
    return p * np.log2(p/q) + (1-p) * np.log2((1-p)/(1-q))


def entropy_level2(prob_bit: ProbabilisticBit) -> float:
    """
    Calculate the entropy of a level 2 probabilistic bit.
    This is just the Shannon entropy.
    
    Args:
        prob_bit: ProbabilisticBit instance
        
    Returns:
        Entropy value
    """
    return shannon_entropy(prob_bit.probability)


def polarization_info_contribution(p: float, s: int) -> float:
    """
    Calculate the informational contribution of polarization in a p-bit.
    
    I_s(p,s) = D_KL(B_p || B_p') where p' is the result of projection to level 2.
    
    Args:
        p: Probability value
        s: Sign/polarization value (+1 or -1)
        
    Returns:
        Information contribution value
    """
    if s == 1:
        return 0.0  # No information loss when s = +1
    else:  # s = -1
        # D_KL(B_p || B_1-p)
        return kl_divergence(p, 1-p)


def entropy_level3(pbit: PBit) -> float:
    """
    Calculate the generalized entropy of a level 3 p-bit.
    
    H_3(p,s) = H(p) + I_s(p,s)
    
    Args:
        pbit: PBit instance
        
    Returns:
        Generalized entropy value
    """
    p = pbit.probability
    s = pbit.sign
    
    # Calculate Shannon entropy component
    h_p = shannon_entropy(p)
    
    # Calculate polarization information contribution
    i_s = polarization_info_contribution(p, s)
    
    return h_p + i_s


def phase_distribution(theta: float, p: float, phi: float) -> float:
    """
    Calculate the phase distribution for a phase state.
    
    f_p(θ,φ) = (1 + 2√(p(1-p))cos(θ-φ))/(2π)
    
    Args:
        theta: Phase angle to evaluate (0 to 2π)
        p: Probability value of the phase state
        phi: Phase of the state
        
    Returns:
        Probability density at the given angle
    """
    return (1 + 2 * np.sqrt(p * (1-p)) * np.cos(theta - phi)) / (2 * np.pi)


def phase_entropy_contribution(p: float, phi: float) -> float:
    """
    Calculate the entropy contribution of the phase component.
    
    S_φ(p,φ) = -∫_0^2π f_p(θ,φ)log_2(f_p(θ,φ)) dθ
    
    Args:
        p: Probability value
        phi: Phase value
        
    Returns:
        Phase entropy contribution
    """
    # Numerically integrate using a dense grid over the relative angle u = θ - φ.
    # This removes any spurious dependence on φ from discretization and respects
    # the analytical φ-invariance of the integral.
    u = np.linspace(0.0, 2.0 * np.pi, 12000, endpoint=False)
    # f_p(u) = (1 + 2√(p(1-p))cos u)/(2π)
    a = 2.0 * np.sqrt(p * (1.0 - p))
    f_vals = (1.0 + a * np.cos(u)) / (2.0 * np.pi)
    # Guard against tiny negative values due to numerical issues
    f_vals = np.clip(f_vals, 1e-15, None)
    integrand = -f_vals * np.log2(f_vals)
    result = np.trapz(integrand, u)
    return float(result)


def entropy_level4(phase_state: PhaseState) -> float:
    """
    Calculate the quasi-quantum entropy of a level 4 phase state.
    
    H_4(p,e^iφ) = H(p) + S_φ(p,φ)
    
    Args:
        phase_state: PhaseState instance
        
    Returns:
        Quasi-quantum entropy value
    """
    p = phase_state.probability
    phi = phase_state.phase
    
    # Calculate Shannon entropy component
    h_p = shannon_entropy(p)
    
    # Calculate phase entropy contribution
    s_phi = phase_entropy_contribution(p, phi)
    
    return h_p + s_phi


def von_neumann_entropy(quantum_state: QuantumState) -> float:
    """
    Calculate the von Neumann entropy of a quantum state.
    
    S(ρ) = -Tr(ρ log₂ ρ) = -Σ λᵢ log₂ λᵢ
    
    where λᵢ are the eigenvalues of the density matrix ρ.
    
    Args:
        quantum_state: QuantumState instance
        
    Returns:
        von Neumann entropy value
    """
    # Build density matrix ρ = |ψ⟩⟨ψ| and compute eigenvalues
    amplitudes = quantum_state.amplitudes  # shape (2,)
    rho = np.outer(amplitudes, np.conjugate(amplitudes))  # 2x2 density matrix
    # Hermitian, use eigvalsh for numerical stability
    eigenvalues = np.linalg.eigvalsh(rho).real
    # Clip small negatives/rounding errors and avoid log(0)
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    mask = eigenvalues > 0.0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(eigenvalues[mask] * np.log2(eigenvalues[mask])))


def calculate_entropy(state: State) -> float:
    """
    Calculate the appropriate entropy for a state at any level of the hierarchy.
    
    Args:
        state: A state from any level of the hierarchy
        
    Returns:
        The appropriate entropy value for the state
    """
    level = state.level
    
    if level == 1:
        # For classical bits, entropy is always 0 (deterministic)
        return 0.0
    elif level == 2:
        # For probabilistic bits, use Shannon entropy
        return entropy_level2(state)
    elif level == 3:
        # For p-bits, use generalized entropy
        return entropy_level3(state)
    elif level == 4:
        # For phase states, use quasi-quantum entropy
        return entropy_level4(state)
    elif level == 5:
        # For quantum states, use von Neumann entropy
        return von_neumann_entropy(state)
    else:
        raise ValueError(f"Unknown state level: {level}")


def information_loss(higher_state: State, lower_state: State) -> float:
    """
    Calculate the information loss when projecting a higher-level state to a lower level.
    
    Args:
        higher_state: The original higher-level state
        lower_state: The projected lower-level state
        
    Returns:
        Information loss value
    """
    if higher_state.level <= lower_state.level:
        raise ValueError("Higher state must have a higher level than lower state")
    
    higher_entropy = calculate_entropy(higher_state)
    lower_entropy = calculate_entropy(lower_state)
    
    return higher_entropy - lower_entropy


def accessible_information(state: State) -> float:
    """
    Calculate the accessible information of a state under classical measurement.
    
    This represents the maximum amount of information that can be extracted
    when projecting the state to level 1 (classical bits).
    
    Args:
        state: A state from any level of the hierarchy
        
    Returns:
        Accessible information value
    """
    level = state.level
    
    if level == 1:
        # For classical bits, all information is accessible
        return 0.0
    
    # Project to level 1 and calculate the Shannon entropy of the result
    projected_state = project(state, 1)
    
    # For classical bits, entropy is 0, so we need to calculate
    # the Shannon entropy of the probability distribution
    if level == 2:
        # For level 2, accessible information is just Shannon entropy
        return entropy_level2(state)
    elif level in [3, 4, 5]:
        # For higher levels, we need to first project to level 2 to get probabilities
        level2_state = project(state, 2)
        return entropy_level2(level2_state)
    else:
        raise ValueError(f"Unknown state level: {level}") 