import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pickle
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

def predict_skin_irritation(smiles):
    """
    Predict skin irritation potential for a molecule using pre-trained model.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        int: Predicted irritation (0 = non-irritant, 1 = irritant)
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
        model_path = os.path.join(os.path.dirname(__file__), 'irritation_forest_descr.pkl')
        with open(model_path, 'rb') as f:
            forest_clf = pickle.load(f)
        logging.info("Successfully loaded skin irritation model")

        df = df.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 128, 130, 12, 14, 17, 19, 20, 21, 22, 23, 131, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 46, 47, 59, 64, 66, 67, 71, 75, 78, 83, 217, 97, 99, 101, 103, 105, 106, 148, 108, 116, 127, 220, 129, 118, 111, 221, 120, 58, 121, 76]]
        # Make prediction (model will handle feature selection)
        prediction = forest_clf.predict(df)[0]
        
        # Return predicted value
        return int(prediction)
    
    except Exception as e:
        logging.error(f"Error predicting skin irritation: {str(e)}")
        return None

def process_dataframe(df, smiles_column):
    """
    Process DataFrame by adding skin irritation predictions for the given SMILES column.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with SMILES
        smiles_column (str): Name of the column containing SMILES strings
        
    Returns:
        pandas.DataFrame: DataFrame with added irritation column
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Add irritation predictions
    result_df['skin_irritation'] = result_df[smiles_column].apply(predict_skin_irritation)
    
    return result_df

if __name__ == "__main__":
    # Example usage with a single SMILES
    test_smiles = "CCO"  # Ethanol
    irritation = predict_skin_irritation(test_smiles)
    if irritation is not None:
        status = "irritant" if irritation == 1 else "non-irritant"
        print(f"Predicted skin irritation for {test_smiles}: {irritation} ({status})")
        
    # Example usage with a DataFrame
    try:
        df = pd.DataFrame({'SMILES': ['CCO', 'CC(=O)NC1=CC=C(O)C=C1', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O']})
        result = process_dataframe(df, 'SMILES')
        print("\nDataFrame results:")
        print(result[['SMILES', 'skin_irritation']])
    except Exception as e:
        print(f"Error processing DataFrame: {str(e)}")
