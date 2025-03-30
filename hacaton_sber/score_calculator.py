import warnings
warnings.filterwarnings("ignore")  # Отключаем все предупреждения

import logging
from hacaton_sber.scoring import calculate_score
from hacaton_sber.Metrics_2.corneal_metric import predict_corneal_permeability
from hacaton_sber.Metrics_2.irritation_metric import predict_skin_irritation
from hacaton_sber.Metrics_2.melanin_metric import predict_melanin_binding
from hacaton_sber.Metrics_2.docking_utils import evaluate_cox2_binding

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
    Рассчитывает скор для заданной молекулы по её SMILES-строке.
    
    Args:
        smiles (str): SMILES-строка молекулы
        
    Returns:
        float or None: Нормализованный скор от 0 до 1 (чем выше, тем лучше), 
                      или None, если невозможно рассчитать скор
    """ 
    # Вычисляем основные метрики для молекулы
    permeability = predict_corneal_permeability(smiles)

    binding = predict_melanin_binding(smiles)

    irritation = predict_skin_irritation(smiles)


    cox2 = evaluate_cox2_binding(smiles, "Compound")
    # Проверяем на наличие None в основных метриках
    if None in [permeability, binding, irritation, cox2]:
        missing = []
        if permeability is None: missing.append("permeability")
        if binding is None: missing.append("melanin binding")
        if irritation is None: missing.append("skin irritation")
        if cox2 is None: missing.append("cox2 binding")
        
        logging.warning(f"Missing required metrics: {', '.join(missing)}")
        
    # Получаем дополнительные метрики
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
    
    # Рассчитываем финальный скор если все необходимые метрики доступны
    if None not in [permeability, binding, irritation, cox2]:
        score = calculate_score(
            permeability=permeability,
            binding=binding,
            irritation=irritation,
            cox2=cox2,
            sascore=sascore if sascore is not None else 10,  # значение по умолчанию если метрика отсутствует
            qed=qed if qed is not None else 0,
            brenk=brenk if brenk is not None else 1,
            binding_energy=general_binding if general_binding is not None else 0
        )
        logging.info(f"Score for SMILES {smiles[:20]}...: {score:.4f}")
        return score
    else:
        logging.warning(f"Unable to calculate score for SMILES {smiles[:20]}... due to missing required metrics")
        return None


if __name__ == "__main__":
    # Пример использования
    test_smiles = "CC(=O)NC1=CC=C(O)C=C1"  # Парацетамол
    score = get_compound_score(test_smiles)
    if score is not None:
        print(f"Скор для молекулы: {score:.4f}")
    else:
        print("Не удалось рассчитать скор для молекулы") 