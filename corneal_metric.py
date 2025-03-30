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
        
        # Select relevant features
        df = df.iloc[:, [130, 41, 107, 11, 14, 83, 217, 25, 58, 3, 2, 4, 1, 0, 63, 207, 147, 80, 137, 87]]
        
        # Make prediction
        prediction = forest_reg.predict(df)[0]
        
        return float(prediction)
    
    except Exception as e:
        print(f"Error predicting corneal permeability: {str(e)}")
        return None
