import warnings
warnings.filterwarnings("ignore")

import logging
import pandas as pd
import argparse
import sys
import os
import random
import time
from pathlib import Path
from score_calculator import get_compound_score, evaluate_compound

# Configure encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_compounds_from_csv(csv_file, smiles_column='SMILES', name_column=None):
    """
    Evaluate a list of compounds from a CSV file.
    
    Args:
        csv_file (str): Path to CSV file with compound list
        smiles_column (str): Name of column containing SMILES strings
        name_column (str): Name of column containing compound names
        
    Returns:
        pandas.DataFrame: Table with evaluation results
    """
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(df)} compounds from {csv_file}")
        
        results = []
        for i, row in df.iterrows():
            smiles = row[smiles_column]
            name = row[name_column] if name_column and name_column in df.columns else f"Compound_{i+1}"
            
            # Evaluate compound
            metrics = evaluate_compound(smiles, name)
            results.append(metrics)
            
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        return results_df
        
    except Exception as e:
        logging.error(f"Error processing file {csv_file}: {str(e)}")
        return None


# Test molecules set
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


def test_batch(output_dir="test_results"):
    """
    Run test batch for a set of well-known molecules.
    
    Args:
        output_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary with molecule names and scores
    """
    logging.info("Starting batch test for standard molecules")
    
    # Create results directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_metrics = []
    
    for name, smiles in test_molecules:
        logging.info(f"Testing molecule: {name}")
        metrics = evaluate_compound(smiles, name)
        all_metrics.append(metrics)
        score = metrics['final_score']
        results[name] = score
        
        if score is not None:
            print(f"Score for {name}: {score:.4f}")
        else:
            print(f"Unable to calculate score for {name}")
    
    # Save all metrics to CSV
    results_df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, "batch_test_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved detailed results to {csv_path}")
    
    return results


def test_rl_batch(output_dir="rl_results"):
    """
    Run test batch for all test molecules simulating an RL cycle.
    
    Simulates a reinforcement learning cycle by evaluating test molecules 
    in a random order with pauses between evaluations.
    
    Args:
        output_dir (str): Directory to save results
    """
    logging.info("Starting RL simulation run for molecules")
    
    # Create results directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Shuffle molecules to simulate RL randomness
    shuffled_molecules = random.sample(test_molecules, len(test_molecules))
    
    # Store run information
    results = []
    
    # Evaluate each molecule
    for i, (name, smiles) in enumerate(shuffled_molecules):
        logging.info(f"[RL Cycle: {i+1}/{len(shuffled_molecules)}] Evaluating molecule: {name}")
        
        start_time = time.time()
        metrics = evaluate_compound(smiles, name)
        elapsed_time = time.time() - start_time
        
        if metrics['final_score'] is not None:
            logging.info(f"Evaluation for {name} completed. Score: {metrics['final_score']:.4f} (in {elapsed_time:.2f} sec)")
            metrics['time'] = elapsed_time
            results.append(metrics)
        else:
            logging.warning(f"Evaluation for {name} failed to produce a score")
            
        # Pause between runs to simulate generation in RL
        if i < len(shuffled_molecules) - 1:
            pause = random.uniform(0.5, 2.0)
            logging.debug(f"Pause between RL cycles: {pause:.2f} sec")
            time.sleep(pause)
    
    # Save summary of results
    if results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "rl_batch_results.csv")
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Saved RL simulation results to {csv_path}")
        
        # Output overall statistics
        valid_scores = [r['final_score'] for r in results if r['final_score'] is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
            
            logging.info(f"=== RL Test Run Summary ===")
            logging.info(f"Total molecules processed: {len(results)}")
            logging.info(f"Average score: {avg_score:.4f}")
            logging.info(f"Best score: {max_score:.4f}")
            logging.info(f"Worst score: {min_score:.4f}")
            
            # Top 3 molecules by score
            top_scores = sorted([(r['name'], r['final_score']) for r in results], key=lambda x: x[1], reverse=True)[:3]
            logging.info(f"Top-3 molecules:")
            for name, score in top_scores:
                logging.info(f"  - {name}: {score:.4f}")
        else:
            logging.warning("No molecules received valid scores")


def main():
    """
    Main function to handle command line arguments and run compound evaluation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Testing and evaluation of drug compounds')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', help='SMILES string to evaluate')
    group.add_argument('--csv', help='Path to CSV file with list of SMILES')
    group.add_argument('--test-batch', action='store_true', help='Run batch test on standard molecules')
    group.add_argument('--test-rl', action='store_true', help='Run reinforcement learning simulation test')
    parser.add_argument('--smiles-column', default='SMILES', help='Name of column with SMILES in CSV (default: SMILES)')
    parser.add_argument('--name-column', help='Name of column with compound names in CSV')
    parser.add_argument('--output', help='Path to save results to CSV')
    parser.add_argument('--output-dir', default='results', help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    if args.smiles:
        # Evaluate single compound
        metrics = evaluate_compound(args.smiles)
        
        # Display results on screen
        print("\nEvaluation Results:")
        print(f"SMILES: {metrics['smiles']}")
        print(f"Corneal Permeability: {metrics['permeability']}")
        print(f"Melanin Binding: {metrics['melanin_binding']}")
        print(f"Skin Irritation: {metrics['skin_irritation']}")
        print(f"COX2 Binding: {metrics['cox2_binding']}")
        print(f"SA Score: {metrics['sascore']}")
        print(f"QED: {metrics['qed']}")
        print(f"Brenk: {metrics['brenk']}")
        print(f"Binding Energy: {metrics['binding_energy']}")
        
        if metrics['final_score'] is not None:
            print(f"Final Score: {metrics['final_score']:.4f}")
        else:
            print("Final score not calculated due to missing required metrics")
    
    elif args.csv:
        # Evaluate compounds from CSV
        results_df = evaluate_compounds_from_csv(args.csv, args.smiles_column, args.name_column)
        
        if results_df is not None:
            # Display overall statistics
            print(f"\nProcessed compounds: {len(results_df)}")
            
            # Sort by final score (if available)
            if 'final_score' in results_df.columns and not results_df['final_score'].isna().all():
                results_df = results_df.sort_values('final_score', ascending=False)
                print(f"Best score: {results_df['final_score'].max():.4f}")
                print(f"Worst score: {results_df['final_score'].min():.4f}")
                print(f"Average score: {results_df['final_score'].mean():.4f}")
            
            # Save results to CSV
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
            else:
                # Display top 5 compounds
                print("\nTop 5 compounds:")
                for i, row in results_df.head(5).iterrows():
                    name = row['name']
                    score = row['final_score']
                    print(f"{name}: {score if score is not None else 'No data'}")
    
    elif args.test_batch:
        # Run batch test
        test_batch(args.output_dir)
    
    elif args.test_rl:
        # Run RL simulation
        test_rl_batch(args.output_dir)


if __name__ == "__main__":
    # Simple usage example
    if len(sys.argv) == 1:
        # If no arguments provided, show a simple example
        print("No arguments provided. Running simple example...")
        test_smiles = "CC(=O)NC1=CC=C(O)C=C1"  # Paracetamol
        print(f"Testing score calculation for Paracetamol...")
        score = get_compound_score(test_smiles)
        if score is not None:
            print(f"Score: {score:.4f}")
        else:
            print("Unable to calculate score")
        print("\nFor more options, run with --help")
    else:
        # Otherwise, process command line arguments
        main() 