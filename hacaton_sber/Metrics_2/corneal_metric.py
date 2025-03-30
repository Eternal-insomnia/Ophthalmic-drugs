import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import os
import logging

def calculate_all_descriptors(smiles):
    """
    Вычисляет все дескрипторы для заданного SMILES.
    Возвращает словарь с дескрипторами или None для каждого дескриптора при ошибке.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            **{name: None for name, _ in Descriptors.descList},
            'logP': None, 'NumAromaticAtoms': None, 'NumAromaticBonds': None,
            'NumHydrophobicAtoms': None, 'NumHydrophilicAtoms': None
        }

    # Вычисление всех дескрипторов из RDKit
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except Exception as e:
            descriptors[name] = None

    # Дополнительные дескрипторы из исходного кода
    try:
        descriptors['logP'] = Crippen.MolLogP(mol)
        descriptors['NumAromaticAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        descriptors['NumAromaticBonds'] = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        descriptors['NumHydrophobicAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [6, 8])  # C и O
        descriptors['NumHydrophilicAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8])  # N и O
    except Exception as e:
        descriptors.update({'logP': None, 'NumAromaticAtoms': None, 'NumAromaticBonds': None,
                            'NumHydrophobicAtoms': None, 'NumHydrophilicAtoms': None})

    return descriptors

def predict_corneal_permeability(smiles):
    """
    Predict corneal permeability for a molecule using pre-trained model.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        float: Predicted corneal permeability value (0-6 scale)
    """
    try:
        # Calculate all descriptors
        desc_dict = calculate_all_descriptors(smiles)
        if all(v is None for v in desc_dict.values()):
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Convert dict to DataFrame
        df = pd.DataFrame([desc_dict])
        
        # Replace None values with NaN
        df = df.replace({None: np.nan})
        
        # Load the trained model from file
        model_path = os.path.join(os.path.dirname(__file__), 'corneal_forest_descr.pkl')
        with open(model_path, 'rb') as f:
            forest_reg = pickle.load(f)
        logging.info("Successfully loaded corneal permeability model")
        df=df.iloc[:, [130, 41, 107, 11, 14, 83, 217, 25, 58, 3, 2, 4, 1, 0, 63, 207, 147, 80, 137, 87]]
        # Make prediction (model will handle feature selection)
        prediction = forest_reg.predict(df)[0]
        
        # Return predicted value
        return float(prediction)
    
    except Exception as e:
        print(f"Error predicting corneal permeability: {str(e)}")
        return None

def process_dataframe(df, smiles_column):
    """
    Process DataFrame by adding corneal permeability predictions for the given SMILES column.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with SMILES
        smiles_column (str): Name of the column containing SMILES strings
        
    Returns:
        pandas.DataFrame: DataFrame with added permeability column
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Add permeability predictions
    result_df['corneal_permeability'] = result_df[smiles_column].apply(predict_corneal_permeability)
    
    return result_df

if __name__ == "__main__":
    # Example usage with a single SMILES
    test_smiles = "CCO"  # Ethanol
    permeability = predict_corneal_permeability(test_smiles)
    if permeability is not None:
        print(f"Predicted corneal permeability for {test_smiles}: {permeability:.2f}")
        
    # Example usage with a DataFrame
    try:
        df = pd.DataFrame({'SMILES': ['CCO', 'CC(=O)NC1=CC=C(O)C=C1', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O']})
        result = process_dataframe(df, 'SMILES')
        print("\nDataFrame results:")
        print(result[['SMILES', 'corneal_permeability']])
    except Exception as e:
        print(f"Error processing DataFrame: {str(e)}")
