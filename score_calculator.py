import warnings
warnings.filterwarnings("ignore")  # Отключаем все предупреждения

import logging
import sys
import os
from scoring import calculate_score
from Metrics_2.corneal_metric import predict_corneal_permeability
from Metrics_2.irritation_metric import predict_skin_irritation
from Metrics_2.melanin_metric import predict_melanin_binding
from Metrics_2.docking_utils import evaluate_cox2_binding

# Configure encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Импорт дополнительных метрик
try:
    from Metrics_2.function_for_neobezatelnie_mectrics_dlya_pluseka import (
        calc_sa_score, calc_qed, calc_brenk, calc_binding_energy
    )
    OPTIONAL_METRICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optional metrics module import error: {e}")
    OPTIONAL_METRICS_AVAILABLE = False
except AttributeError as e:
    logging.warning(f"Optional metrics attribute error: {e}")
    OPTIONAL_METRICS_AVAILABLE = False


def get_compound_score(smiles):
    """
    Calculate normalized score for a molecule from its SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        float or None: Normalized score from 0 to 1 (higher is better),
                      or None if unable to calculate the score
    """
    # Calculate primary metrics
    permeability = predict_corneal_permeability(smiles)
    binding = predict_melanin_binding(smiles)
    irritation = predict_skin_irritation(smiles)
    cox2 = evaluate_cox2_binding(smiles, "Compound")
    
    # Check for missing primary metrics
    if None in [permeability, binding, irritation, cox2]:
        missing = []
        if permeability is None: missing.append("permeability")
        if binding is None: missing.append("melanin binding")
        if irritation is None: missing.append("skin irritation")
        if cox2 is None: missing.append("cox2 binding")
        
        logging.warning(f"Missing required metrics: {', '.join(missing)}")
        
    # Get optional metrics
    sascore = None
    qed = None
    brenk = None
    general_binding = None
    
    if OPTIONAL_METRICS_AVAILABLE:
        try:
            sascore = calc_sa_score(smiles)
        except Exception as e:
            logging.debug(f"Error calculating SA score: {e}")
            
        try:
            qed = calc_qed(smiles)
        except Exception as e:
            logging.debug(f"Error calculating QED: {e}")
            
        try:
            brenk = calc_brenk(smiles)
        except Exception as e:
            logging.debug(f"Error calculating Brenk: {e}")
            
        try:
            general_binding = calc_binding_energy(smiles)
        except Exception as e:
            logging.debug(f"Error calculating binding energy: {e}")
    
    # Calculate final score if all required metrics are available
    if None not in [permeability, binding, irritation, cox2]:
        score = calculate_score(
            permeability=permeability,
            binding=binding,
            irritation=irritation,
            cox2=cox2,
            sascore=sascore if sascore is not None else 10,  # default value if metric is missing
            qed=qed if qed is not None else 0,
            brenk=brenk if brenk is not None else 1,
            binding_energy=general_binding if general_binding is not None else 0
        )
        logging.info(f"Score for SMILES {smiles[:20]}...: {score:.4f}")
        return score
    else:
        logging.warning(f"Unable to calculate score for SMILES {smiles[:20]}... due to missing required metrics")
        return None


def evaluate_compound(smiles, name="Compound"):
    """
    Evaluate a compound using all available metrics and return detailed results.
    
    Args:
        smiles (str): SMILES string of the molecule
        name (str): Name of the compound for output
        
    Returns:
        dict: Dictionary with all metric results and final score
    """
    logging.info(f"Evaluating compound: {name} ({smiles})")
    
    # Get primary metrics from models
    permeability = predict_corneal_permeability(smiles)
    binding = predict_melanin_binding(smiles)
    irritation = predict_skin_irritation(smiles)
    cox2 = evaluate_cox2_binding(smiles, name)
    
    # Check for missing primary metrics
    if None in [permeability, binding, irritation, cox2]:
        missing = []
        if permeability is None: missing.append("permeability")
        if binding is None: missing.append("melanin binding")
        if irritation is None: missing.append("skin irritation")
        if cox2 is None: missing.append("cox2 binding")
        
        logging.warning(f"Missing required metrics for {name}: {', '.join(missing)}")
        
    # Get optional metrics with error handling
    sascore = None
    qed = None
    brenk = None
    general_binding = None
    
    if OPTIONAL_METRICS_AVAILABLE:
        try:
            sascore = calc_sa_score(smiles)
        except Exception as e:
            logging.warning(f"Error calculating SA score: {e}")
            
        try:
            qed = calc_qed(smiles)
        except Exception as e:
            logging.warning(f"Error calculating QED: {e}")
            
        try:
            brenk = calc_brenk(smiles)
        except Exception as e:
            logging.warning(f"Error calculating Brenk: {e}")
            
        try:
            general_binding = calc_binding_energy(smiles)
        except Exception as e:
            logging.warning(f"Error calculating binding energy: {e}")
    else:
        logging.warning("Optional metrics module not available, using default values")
    
    # Collect all metrics in a dictionary
    metrics = {
        'smiles': smiles,
        'name': name,
        'permeability': permeability,
        'melanin_binding': binding,
        'skin_irritation': irritation,
        'cox2_binding': cox2,
        'sascore': sascore,
        'qed': qed,
        'brenk': brenk,
        'binding_energy': general_binding
    }
    
    # Calculate final score if all required metrics are available
    if None not in [permeability, binding, irritation, cox2]:
        score = calculate_score(
            permeability=permeability,
            binding=binding,
            irritation=irritation,
            cox2=cox2,
            sascore=sascore if sascore is not None else 10,
            qed=qed if qed is not None else 0,
            brenk=brenk if brenk is not None else 1,
            binding_energy=general_binding if general_binding is not None else 0
        )
        metrics['final_score'] = score
        logging.info(f"Final score for {name}: {score:.4f}")
    else:
        metrics['final_score'] = None
        logging.warning(f"Unable to calculate final score for {name} due to missing required metrics")
    
    return metrics 