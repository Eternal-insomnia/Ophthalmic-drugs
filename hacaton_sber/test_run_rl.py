import subprocess
import logging
import pandas as pd
import random
import time
import sys
import os
from pathlib import Path

# Настройка кодировки для Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_testing_real.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Набор известных молекул для тестирования
test_molecules = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Paracetamol", "CC(=O)NC1=CC=C(O)C=C1"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Diclofenac", "O=C(O)CC1=CC=CC=C1NC2=C(Cl)C=C(Cl)C=C2"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Propranolol", "CC(C)NCC(O)COC1=CC=C(C=C1)CC2=CC=CC=C2"),
    ("Atropine", "CN1C2CCC1CC(C2)OC(=O)C(CO)C3=CC=CC=C3"),
    ("Warfarin", "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O"),
    ("Lidocaine", "CCN(CC)CC(=O)NC1=C(C)C=CC=C1C")
]

def run_main_for_smiles(smiles, name, output_dir="rl_results_real"):
    """
    Execute main.py to evaluate a given SMILES
    
    Args:
        smiles (str): SMILES string to evaluate
        name (str): Molecule name for logging
        output_dir (str): Directory for saving results
    
    Returns:
        dict: Dictionary with results or None on error
    """
    # Create results directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare command - use real models
    output_file = f"{output_dir}/{name.replace(' ', '_')}_result.csv"
    cmd = [
        "python", "main.py", 
        "--smiles", smiles
    ]
    
    logging.info(f"Starting evaluation with REAL MODELS for {name}: {smiles}")
    start_time = time.time()
    
    try:
        # Run process
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Calculate execution time
        elapsed_time = time.time() - start_time
        
        # Extract results from output
        output = result.stdout
        logging.info(f"Main.py stdout: {output}")
        
        # Log stderr if any
        if result.stderr:
            logging.info(f"Main.py stderr: {result.stderr}")
        
        # Extract final score from output
        final_score = None
        for line in output.split('\n'):
            if "final score" in line.lower() or "final_score" in line.lower():
                try:
                    final_score = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
        
        # Log result
        if final_score is not None:
            logging.info(f"Evaluation for {name} completed successfully. Score: {final_score:.2f} (in {elapsed_time:.2f} sec)")
            return {"name": name, "smiles": smiles, "score": final_score, "time": elapsed_time}
        else:
            logging.warning(f"Evaluation for {name} completed, but no score found (in {elapsed_time:.2f} sec)")
            # Проверяем детали метрик
            metrics = {}
            for line in output.split('\n'):
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metrics[key] = value
            logging.info(f"Metrics extracted from output: {metrics}")
            return {"name": name, "smiles": smiles, "score": None, "time": elapsed_time}
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error evaluating {name}: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logging.error(f"Error details: {e.stderr}")  # Show full error
        return None
    except Exception as e:
        logging.error(f"Unexpected error evaluating {name}: {str(e)}")
        return None

def test_rl_batch():
    """Run test batch for all test molecules using real models"""
    logging.info("Starting REAL MODEL run for 10 molecules (RL cycle simulation)")
    
    # Shuffle molecules to simulate RL randomness
    shuffled_molecules = random.sample(test_molecules, len(test_molecules))
    
    # Store run information
    results = []
    
    # Evaluate each molecule
    for i, (name, smiles) in enumerate(shuffled_molecules):
        logging.info(f"[RL Cycle: {i+1}/10] Evaluating molecule: {name}")
        
        result = run_main_for_smiles(smiles, name)
        if result:
            results.append(result)
            
        # Pause between runs to simulate generation in RL
        if i < len(shuffled_molecules) - 1:
            pause = random.uniform(0.5, 2.0)
            logging.debug(f"Pause between RL cycles: {pause:.2f} sec")
            time.sleep(pause)
    
    # Save summary of results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("rl_batch_results_real.csv", index=False)
        
        # Output overall statistics
        valid_scores = results_df[results_df['score'].notna()]['score']
        if not valid_scores.empty:
            logging.info(f"=== REAL MODEL RL Test Run Summary ===")
            logging.info(f"Total molecules processed: {len(results)}")
            logging.info(f"Molecules with valid scores: {len(valid_scores)}")
            logging.info(f"Average score: {valid_scores.mean():.2f}")
            logging.info(f"Best score: {valid_scores.max():.2f} ({results_df.loc[valid_scores.idxmax()]['name']})")
            logging.info(f"Worst score: {valid_scores.min():.2f} ({results_df.loc[valid_scores.idxmin()]['name']})")
            logging.info(f"Average evaluation time: {results_df['time'].mean():.2f} sec")
            
            # Sort by score for report
            top_results = results_df.sort_values('score', ascending=False).head(3)
            logging.info(f"Top-3 molecules:")
            for _, row in top_results.iterrows():
                logging.info(f"  - {row['name']}: {row['score']:.2f}")
        else:
            logging.warning("No molecules received valid scores")

if __name__ == "__main__":
    logging.info("Starting RL simulation test run with REAL MODELS")
    test_rl_batch()
    logging.info("Real models test run completed") 