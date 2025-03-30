from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import AllChem
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calc_sa_score(smiles):
    """
    Вычисляет SAscore для молекулы SMILES.
    SAscore - это синтетическая доступность (меньше значение = легче синтезировать).
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: значение SAscore (обычно от 1 до 10)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def calc_qed(smiles):
    """
    Вычисляет QED (Quantitative Estimation of Drug-likeness) для молекулы SMILES.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: значение QED (от 0 до 1, чем выше тем лучше)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)

def calc_brenk(smiles):
    """
    Вычисляет количество нарушений правил Бренка для молекулы SMILES.
    Правила Бренка - это набор структурных фрагментов, которые могут вызывать проблемы с абсорбцией,
    распределением, метаболизмом, выведением или токсичностью.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: нормализованное количество нарушений (от 0 до 1)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    # Создаем каталог фильтров для правил Бренка
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    
    # Получаем количество срабатываний фильтров Бренка
    matches = catalog.GetMatches(mol)
    num_matches = len(matches)
    
    # Нормализуем значение (примерное значение, можно настроить)
    # Обычно делается деление на максимальное ожидаемое количество нарушений
    max_expected_violations = 5.0
    normalized_value = min(num_matches / max_expected_violations, 1.0)
    
    return normalized_value

def calc_binding_energy(smiles):
    """
    Вычисляет приближенную энергию связывания молекулы.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: оценка энергии связывания (отрицательные значения лучше)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    try:
        mol = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if embed_status == -1:
            logger.debug(f"Failed to embed molecule: {smiles}")
            return None
            
        # Оптимизация с перехватом ошибок
        opt_status = AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        if opt_status == -1:
            logger.debug(f"Failed to optimize molecule: {smiles}")
            return None
            
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.MolWt(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        # Улучшенная формула с учетом дополнительных параметров
        binding_energy = (-0.4 * logp 
                          - 0.1 * tpsa / 100 
                          - 0.05 * molwt / 100 
                          - 0.15 * rotatable_bonds
                          - 0.2 * h_donors
                          - 0.15 * h_acceptors)
        
        return binding_energy
        
    except Exception as e:
        logger.debug(f"Error in binding energy calculation for {smiles}: {str(e)}")
        return None

def process_single_molecule(smiles):
    """
    Обрабатывает одну молекулу и возвращает словарь с метриками.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        dict: словарь с метриками или None при ошибке
    """
    # Проверка валидности SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Вычисление метрик
    sa_score = calc_sa_score(smiles)
    qed_score = calc_qed(smiles)
    brenk_score = calc_brenk(smiles)
    
    try:
        binding_energy = calc_binding_energy(smiles)
    except Exception as e:
        logger.debug(f"Error calculating binding energy for {smiles}: {str(e)}")
        binding_energy = None
    
    # Проверка на None
    if any(x is None for x in [sa_score, qed_score, brenk_score, binding_energy]):
        return None
    
    return {
        'SMILES': smiles,
        'SAscore': sa_score,
        'QED': qed_score,
        'Brenk': brenk_score,
        'BindingEnergy': binding_energy
    }

def filter_and_sort_molecules(smiles_list, parallel=True, thresholds=None):
    """
    Фильтрует и сортирует молекулы по заданным критериям.
    
    Args:
        smiles_list: список SMILES-строк молекул
        parallel: использовать ли параллельную обработку (True/False)
        thresholds: словарь с пороговыми значениями для фильтрации
    
    Returns:
        list: отсортированный список словарей с молекулами и их метриками
    """
    logger.info(f"Processing {len(smiles_list)} molecules")
    
    # Настройка порогов фильтрации
    if thresholds is None:
        thresholds = {
            'SAscore_max': 3.0,
            'QED_min': 0.6,
            'Brenk_max': 0.4,
            'BindingEnergy_min': -12.0,
            'BindingEnergy_max': -6.0
        }
    
    # Обработка молекул (последовательно или параллельно)
    if parallel and len(smiles_list) > 10:
        # Параллельная обработка
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using parallel processing with {n_cores} cores")
        
        with multiprocessing.Pool(processes=n_cores) as pool:
            molecules = list(tqdm(
                pool.imap(process_single_molecule, smiles_list),
                total=len(smiles_list),
                desc="Processing molecules"
            ))
    else:
        # Последовательная обработка
        molecules = []
        for smiles in tqdm(smiles_list, desc="Processing molecules"):
            result = process_single_molecule(smiles)
            if result:
                molecules.append(result)
    
    # Удаляем None значения (неуспешные обработки)
    molecules = [m for m in molecules if m is not None]
    
    logger.info(f"Successfully processed {len(molecules)} valid molecules")
    
    # Фильтрация молекул по заданным порогам
    filtered_molecules = [
        mol for mol in molecules
        if (mol['SAscore'] < thresholds['SAscore_max'] and
            mol['QED'] > thresholds['QED_min'] and
            mol['Brenk'] < thresholds['Brenk_max'] and
            thresholds['BindingEnergy_min'] <= mol['BindingEnergy'] < thresholds['BindingEnergy_max'])
    ]
    
    logger.info(f"{len(filtered_molecules)} molecules passed all filters")
    
    # Сортировка молекул по метрикам
    sorted_molecules = sorted(
        filtered_molecules,
        key=lambda x: (x['SAscore'], -x['QED'], x['Brenk'], x['BindingEnergy'])
    )
    
    return sorted_molecules

def save_results(molecules, output_file='filtered_molecules.csv'):
    """
    Сохраняет результаты в CSV файл.
    
    Args:
        molecules: список словарей с молекулами и их метриками
        output_file: имя файла для сохранения результатов
    """
    if not molecules:
        logger.warning("No molecules to save")
        return
        
    df = pd.DataFrame(molecules)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    return df

# Пример использования
if __name__ == "__main__":
    import sys
    
    # Чтение SMILES из файла или аргументов командной строки
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            with open(input_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file {input_file}: {str(e)}")
            sys.exit(1)
    else:
        # Тестовые примеры
        smiles_list = [
            "CCO",                  # Этанол
            "c1ccccc1",             # Бензол
            "CC(=O)OC1=CC=CC=C1C(=O)O", # Аспирин
            "CCN(CC)CC",            # Триэтиламин
            "CC1=C(C(=O)C2=C(C1=O)N3CC4C(C3)C(=O)N4C5=CC=CC=C5)O" # Недопустимая
        ]
    
    # Обработка молекул
    ranked_molecules = filter_and_sort_molecules(smiles_list)
    
    # Сохранение и вывод результатов
    df = save_results(ranked_molecules)
    if not df.empty:
        print("\nТоп 5 молекул:")
        print(df.head().to_string(index=False))