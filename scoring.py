"""
Scoring module for evaluating compound properties.

This module contains functions for calculating a composite score
based on multiple pharmacological and physicochemical properties.
"""


def calculate_score(
    permeability: float,  # 0 to 6, maximize
    binding: int,  # 0 or 1, prefer 1
    irritation: int,  # 0 or 1, prefer 0
    cox2: float,  # -12 to 0, minimize
    sascore: float,  # minimize
    qed: float,  # 0 to 1, maximize
    brenk: float,  # minimize
    binding_energy: float  # minimize, < -6 preferred
) -> float:
    """
    Calculate compound score based on multiple properties.
    
    Args:
        permeability: Permeability score (0-6, higher is better)
        binding: Target binding (0 or 1, 1 is preferred)
        irritation: Skin irritation potential (0 or 1, 0 is preferred)
        cox2: COX2 binding energy (-12 to 0, lower is better)
        sascore: Synthetic accessibility score (lower is better)
        qed: Quantitative Estimation of Drug-likeness (0-1, >0.6 preferred)
        brenk: Brenk structural alerts score (lower is better, <0.4 preferred)
        binding_energy: General binding energy (lower is better, <-6 preferred)
        
    Returns:
        float: Normalized composite score (0-1, higher is better)
    """
    # Calculate important metrics score components
    if permeability > 4:
        residue = permeability - 4
        if residue > 2:
            residue = 2
        permeability = 7.5 + residue * 1.25  # reward 7.5 for >4, up to 10 for 6
    else:
        permeability = 0

    if binding == 1: 
        binding = 10
    else: 
        binding = 0
    
    if irritation == 0: 
        irritation = 10
    else: 
        irritation = 0

    if cox2 <= -6 and cox2 >= -12: 
        cox2 = 7.5 + 2.5 - (cox2 + 12) * (2.5 / 6)
    else: 
        cox2 = 0

    important_metrics = permeability + binding + irritation + cox2

    # Calculate less important metrics score components
    if sascore <= 3 and sascore >= 1: 
        sascore = 0.75 + (0.25 - (sascore - 1) * (0.25 / 2))
    else: 
        sascore = 0
    
    if qed >= 0.6 and qed <= 10: 
        qed = 0.75 + ((qed - 0.6) / (1 - 0.6)) * 0.25
    else: 
        qed = 0

    if brenk <= 0.4 and brenk >= 0: 
        brenk = 0.75 + 0.25 * (1 - brenk / 0.4)
    else: 
        brenk = 0

    if binding_energy <= -6 and binding_energy >= -12: 
        binding_energy = 0.75 + 0.25 - (binding_energy + 12) * (0.25 / 6)
    else: 
        binding_energy = 0

    unimportant_metrics = sascore + qed + brenk + binding_energy

    # Calculate final raw score
    raw_score = important_metrics + unimportant_metrics
    
    # Normalization to 0-1 range
    # Maximum possible score calculation:
    # - Important metrics: 10 (permeability) + 10 (binding) + 10 (irritation) + 10 (cox2) = 40
    # - Less important metrics: 1 (sascore) + 1 (qed) + 1 (brenk) + 1 (binding_energy) = 4
    # - Total: approximately 44 (but actual maximum closer to 48)
    MAX_POSSIBLE_SCORE = 48.0
    
    # Normalize score to 0-1 range
    normalized_score = max(0.0, min(1.0, raw_score / MAX_POSSIBLE_SCORE))
    
    return normalized_score 