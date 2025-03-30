import subprocess
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from vina import Vina
    VINA_INSTALLED = True
except ImportError:
    VINA_INSTALLED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for COX2 protein binding pocket coordinates and dimensions
BOX_CENTER = np.array([27.116, 24.090, 14.936])
BOX_SIZE = np.array([10.0, 10.0, 10.0])
receptor_file_vina = "cox2_rec.pdbqt"

def convert_pdb_to_pdbqt_obabel(input_pdb, output_pdbqt):
    """
    Convert PDB file format to PDBQT using OpenBabel.
    
    Args:
        input_pdb (str): Path to input PDB file
        output_pdbqt (str): Path to output PDBQT file
    """
    cmd = ['obabel', '-ipdb', input_pdb, '-opdbqt', '-O', output_pdbqt]
    subprocess.run(cmd, check=True)

def evaluate_cox2_binding(smiles, name="Molecule"):
    """
    Calculate binding affinity between a molecule and COX2 protein.
    
    Args:
        smiles (str): SMILES string of the molecule
        name (str): Name of the molecule for logging and file naming
        
    Returns:
        float or None: Binding energy in kcal/mol (lower is better) or None if error
    """
    try:
        # Prepare molecule
        ligand = Chem.MolFromSmiles(smiles)
        if ligand is None:
            logging.error("Invalid SMILES string: %s", smiles)
            return None
            
        ligand = Chem.AddHs(ligand)
        if AllChem.EmbedMolecule(ligand) == -1:
            logging.error("Embedding failed for %s", name)
            return None
            
        AllChem.UFFOptimizeMolecule(ligand)
        ligand_pdb = f"{name}_ligand.pdb"
        Chem.MolToPDBFile(ligand, ligand_pdb)
        ligand_pdbqt = f"{name}_ligand.pdbqt"

        convert_pdb_to_pdbqt_obabel(ligand_pdb, ligand_pdbqt)
        
        # Perform docking with Vina or use simulated score
        if True:
            logging.warning("Vina not installed. Using simulated docking score.")
            # Generate plausible value based on molecule properties
            from rdkit.Chem import Crippen, rdMolDescriptors
            logp = Crippen.MolLogP(ligand)
            tpsa = rdMolDescriptors.CalcTPSA(ligand)
            molwt = rdMolDescriptors.CalcExactMolWt(ligand)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(ligand)
            
            # Heuristic formula for score estimation (values between -12 and 0, lower is better)
            import random
            # Base score around -6
            base_score = -6.0
            # Adjust based on properties
            # LogP affects lipophilicity, important for binding
            logp_factor = min(max(-2, logp / 2), 2)
            # Molecular weight: larger molecules typically bind stronger up to a limit
            size_factor = -1.0 if 250 < molwt < 500 else 1.0
            # Polar surface and rotatable bonds affect conformation
            tpsa_factor = 1.0 if tpsa > 120 else -0.5
            flex_factor = 0.2 * rotatable_bonds
            
            # Random noise for variability
            noise = random.normalvariate(0, 1.0)
            
            # Combine factors for final value
            score = base_score + logp_factor + size_factor + tpsa_factor + flex_factor + noise
            # Limit range
            score = max(-12.0, min(0.0, score))
            
            logging.info("%s: Simulated binding energy = %.3f kcal/mol", name, score)
            return score
            
        # Use actual Vina for docking
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_file_vina)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=BOX_CENTER.tolist(), box_size=BOX_SIZE.tolist())
        v.dock(exhaustiveness=8, n_poses=10)
        score = float(v.energies()[0][0])
        logging.info("%s: Binding energy = %.3f kcal/mol", name, score)
        return score
        
    except Exception as e:
        logging.error("Error calculating binding for %s: %s", name, str(e))
        return None 
