from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Crippen, BRICS, AllChem, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED

def calc_sa_score(smiles):
    """
    Вычисляет SAscore (синтетическую доступность) для молекулы SMILES.
    Меньшее значение = легче синтезировать (диапазон обычно от 1 до 10).
    
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
    Правила Бренка - это набор структурных фрагментов, которые могут вызывать 
    проблемы с ADMET свойствами.
    
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
    catalog = FilterCatalog.FilterCatalog(params)
    
    # Получаем количество срабатываний фильтров Бренка
    matches = catalog.GetMatches(mol)
    num_matches = len(matches)
    
    # Нормализуем значение
    max_expected_violations = 5.0
    normalized_value = min(num_matches / max_expected_violations, 1.0)
    
    return normalized_value

def calc_binding_energy(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)  # Здесь возникает RuntimeError
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    molwt = Descriptors.MolWt(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Упрощенная эвристическая формула (условная, не для реального применения)
    # Более отрицательное значение = лучшее связывание
    binding_energy = -0.5 * logp - 0.1 * tpsa / 100 - 0.05 * molwt / 100 - 0.2 * rotatable_bonds
    
    return binding_energy


def calc_tpsa(smiles):
    """
    Вычисляет топологическую полярную площадь поверхности (TPSA) молекулы SMILES.
    Для офтальмологических препаратов обычно требуется TPSA < 140 Å².
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: значение TPSA в Å²
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return rdMolDescriptors.CalcTPSA(mol)

def calc_tanimoto_similarity(smiles, reference_smiles):
    """
    Вычисляет схожесть Танимото между молекулой SMILES и референсной молекулой.
    Использует Morgan fingerprints с радиусом 2.
    
    Args:
        smiles: SMILES-строка исследуемой молекулы
        reference_smiles: SMILES-строка референсной молекулы (например, известного препарата)
    Returns:
        float: значение схожести Танимото (от 0 до 1)
    """
    mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    
    if mol is None or ref_mol is None:
        return None
    
    # Morgan fingerprints с радиусом 2
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def calc_logp(smiles):
    """
    Вычисляет LogP (коэффициент распределения октанол/вода) для молекулы SMILES.
    Для офтальмологических капель оптимальный LogP обычно 1-3.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: значение LogP
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Crippen.MolLogP(mol)

def calc_rotatable_bonds(smiles):
    """
    Вычисляет количество вращаемых связей в молекуле SMILES.
    Меньшее количество вращаемых связей (<10) указывает на лучшую биодоступность.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        int: количество вращаемых связей
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return rdMolDescriptors.CalcNumRotatableBonds(mol)

def check_pains(smiles):
    """
    Проверяет молекулу SMILES на наличие PAINS (Pan Assay Interference Compounds).
    PAINS - это структуры, которые могут давать ложноположительные результаты в тестах.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        bool: содержит ли молекула PAINS-структуры
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    
    return catalog.HasMatch(mol)

def calc_molecular_weight(smiles):
    """
    Вычисляет молекулярную массу соединения SMILES.
    Для капель оптимальная молекулярная масса < 500 Да.
    
    Args:
        smiles: SMILES-строка молекулы
    Returns:
        float: молекулярная масса в Да
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return rdMolDescriptors.CalcExactMolWt(mol)

def calculate_score(
    permeability: float,  # 0 to 6, maximize
    binding: int,  # 0 or 1, prefer 1
    irritation: int,  # 0 or 1, prefer 0
    cox2: float,  # -12 to 0, minimize
    sascore: float,  # minimize, <3 preferred
    qed: float,  # 0 to 1, minimize, <0.6 preferred
    brenk: float,  # minimize, <0.4 preferred
    binding_energy: float  # minimize, < -6 preferred
) -> float:
    """
    Вычисляет нормализованный (от 0 до 1) счет для молекулы на основе важных метрик.
    
    Args:
        permeability: проницаемость через роговицу (0-6, лучше выше)
        binding: связывание (0 или 1, предпочтительно 1)
        irritation: раздражение (0 или 1, предпочтительно 0)
        cox2: активность COX-2 (-12 до 0, предпочтительно ближе к -12)
        sascore: синтетическая доступность (1-10, предпочтительно <3)
        qed: drug-likeness (0-1, предпочтительно >0.6)
        brenk: нарушения правил Бренка (0-1, предпочтительно <0.4)
        binding_energy: энергия связывания (предпочтительно < -6)
    
    Returns:
        float: нормализованный счет от 0 до 1
    """
    if permeability > 4:
        residue = permeability - 4
        if residue > 2:
            residue = 2
        permeability = 7.5 + residue * 1.25  # награда 7.5 за то, что угадал больше 4, до 10 если 6
    else:
        permeability = 0

    if binding == 1: binding = 10
    else: binding = 0
    
    if irritation == 0: irritation = 10
    else: irritation = 0

    if cox2 <= -6 and cox2 >= -12: 
        cox2 = 7.5 + 2.5 - (cox2 + 12) * (2.5 / 6)
    else: cox2 = 0

    important_metrics = permeability + binding + irritation + cox2

    if sascore <= 3 and sascore >= 1: sascore = 0.75 + (0.25 - (sascore - 1) * (0.25 / 2))
    else: sascore = 0
    
    if qed >= 0.6 and qed <= 10: qed = 0.75 + ((qed - 0.6) / (1 - 0.6)) * 0.25
    else: qed = 0

    if brenk <= 0.4 and brenk >= 0: brenk = 7.5 + 2.5 * (1 - brenk / 0.4)
    else: brenk = 0

    if binding_energy <= -6 and binding_energy >= -12: 
        binding_energy = 0.75 + 0.25 - (binding_energy + 12) * (0.25 / 6)
    else: binding_energy = 0

    un