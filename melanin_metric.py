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
    Calculate all descriptors for a given SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        dict: Dictionary with descriptors or None for each descriptor on error
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            **{name: None for name, _ in Descriptors.descList},
            'logP': None, 'NumAromaticAtoms': None, 'NumAromaticBonds': None,
            'NumHydrophobicAtoms': None, 'NumHydrophilicAtoms': None
        }

    # Calculate all descriptors from RDKit
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except Exception:
            descriptors[name] = None

    # Calculate additional descriptors
    try:
        descriptors['logP'] = Crippen.MolLogP(mol)
        descriptors['NumAromaticAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        descriptors['NumAromaticBonds'] = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        descriptors['NumHydrophobicAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [6, 8])  # C and O
        descriptors['NumHydrophilicAtoms'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8])  # N and O
    except Exception:
        descriptors.update({'logP': None, 'NumAromaticAtoms': None, 'NumAromaticBonds': None,
                            'NumHydrophobicAtoms': None, 'NumHydrophilicAtoms': None})

    return descriptors

def predict_melanin_binding(smiles):
    """
    Predict melanin binding for a molecule using pre-trained model.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        int: Predicted binding (0 = non-binding, 1 = binding)
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
        model_path = os.path.join(os.path.dirname(__file__), 'melanin_forest_descr.pkl')
        with open(model_path, 'rb') as f:
            forest_clf = pickle.load(f)
        logging.info("Successfully loaded melanin binding model")

        # Select relevant features
        df = df.iloc[:, [2, 67, 3, 37, 99, 131, 59, 81, 20, 89, 218, 219, 28, 62, 129, 115, 32, 106, 46, 33, 34, 30, 31, 121, 41, 29, 9, 35, 7, 8, 6, 217, 130, 36, 220, 38, 205, 58, 39, 4]]

        # Make prediction
        prediction = forest_clf.predict(df)[0]
        
        return int(prediction)
    
    except Exception as e:
        logging.error(f"Error predicting melanin binding: {str(e)}")
        return None
