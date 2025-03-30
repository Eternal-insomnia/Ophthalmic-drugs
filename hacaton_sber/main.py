import warnings
warnings.filterwarnings("ignore")  # Отключаем все предупреждения
import pandas as pd
import logging
import argparse
import sys
import os
import random
from scoring import calculate_score
from Metrics_2.corneal_metric import predict_corneal_permeability
from Metrics_2.irritation_metric import predict_skin_irritation
from Metrics_2.melanin_metric import predict_melanin_binding
from Metrics_2.docking_utils import evaluate_cox2_binding

# Configure encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Proper import of metrics with error handling
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_compound(smiles, name="Compound", test_mode=False):
    """
    Evaluate a compound using all available metrics and calculate final score.
    
    Args:
        smiles (str): SMILES string of the molecule
        name (str): Name of the compound for output
        test_mode (bool): If True, use mock functions instead of real models
        
    Returns:
        dict: Dictionary with all metric results and final score
    """
    logging.info(f"Evaluating compound: {name} ({smiles})")
    
    # Get primary metrics from real models
    permeability = predict_corneal_permeability(smiles)
    binding = predict_melanin_binding(smiles)
    irritation = predict_skin_irritation(smiles)
    cox2 = evaluate_cox2_binding(smiles, name)
    
    # Check for None values in primary metrics
    if None in [permeability, binding, irritation, cox2]:
        missing = []
        if permeability is None: missing.append("permeability")
        if binding is None: missing.append("melanin binding")
        if irritation is None: missing.append("skin irritation")
        if cox2 is None: missing.append("cox2 binding")
        
        logging.warning(f"Missing required metrics for {name}: {', '.join(missing)}")
        
    # Get additional metrics with error handling
    sascore = None
    qed = None
    brenk = None
    general_binding = None
    
    if OPTIONAL_METRICS_AVAILABLE:
        try:
            sascore = calc_sa_score(smiles)
        except Exception as e:
            logging.warning(f"Error calculating SA score: {e}")
            
        try:
            qed = calc_qed(smiles)
        except Exception as e:
            logging.warning(f"Error calculating QED: {e}")
            
        try:
            brenk = calc_brenk(smiles)
        except Exception as e:
            logging.warning(f"Error calculating Brenk: {e}")
            
        try:
            general_binding = calc_binding_energy(smiles)
        except Exception as e:
            logging.warning(f"Error calculating binding energy: {e}")
    else:
        logging.warning("Optional metrics module not available, using default values")
    
    # Collect all metrics in a dictionary
    metrics = {
        'smiles': smiles,
        'name': name,
        'permeability': permeability,
        'melanin_binding': binding,
        'skin_irritation': irritation,
        'cox2_binding': cox2,
        'sascore': sascore,
        'qed': qed,
        'brenk': brenk,
        'binding_energy': general_binding
    }
    
    # Calculate final score if all required metrics are available
    if None not in [permeability, binding, irritation, cox2]:
        score = calculate_score(
            permeability=permeability,
            binding=binding,
            irritation=irritation,
            cox2=cox2,
            sascore=sascore if sascore is not None else 10,  # default value if metric is missing
            qed=qed if qed is not None else 0,
            brenk=brenk if brenk is not None else 1,
            binding_energy=general_binding if general_binding is not None else 0
        )
        metrics['final_score'] = score
        logging.info(f"Final score for {name}: {score:.4f}")
    else:
        metrics['final_score'] = None
        logging.warning(f"Unable to calculate final score for {name} due to missing required metrics")
    
    return metrics

def evaluate_compounds_from_csv(csv_file, smiles_column='SMILES', name_column=None, test_mode=False):
    """
    Evaluate a list of compounds from a CSV file
    
    Args:
        csv_file (str): Path to CSV file with compound list
        smiles_column (str): Name of column containing SMILES strings
        name_column (str): Name of column containing compound names (optional)
        test_mode (bool): If True, use mock functions instead of real models
        
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
            metrics = evaluate_compound(smiles, name, test_mode)
            results.append(metrics)
            
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        return results_df
        
    except Exception as e:
        logging.error(f"Error processing file {csv_file}: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluation of drug compound properties')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', help='SMILES string to evaluate')
    group.add_argument('--csv', help='Path to CSV file with list of SMILES')
    parser.add_argument('--smiles-column', default='SMILES', help='Name of column with SMILES in CSV (default: SMILES)')
    parser.add_argument('--name-column', help='Name of column with compound names in CSV')
    parser.add_argument('--output', help='Path to save results to CSV')
    parser.add_argument('--test', action='store_true', help='Run in test mode with mock data (no models required)')
    
    args = parser.parse_args()
    
    # Check if running in test mode
    if args.test:
        logging.info("Running in TEST MODE with mock data (no models required)")
    
    if args.smiles:
        # Evaluate single compound
        metrics = evaluate_compound(args.smiles, test_mode=args.test)
        
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
            print(f"Final Score: {metrics['final_score']:.2f}")
        else:
            print("Final score not calculated due to missing required metrics")
    
    elif args.csv:
        # Evaluate compounds from CSV
        results_df = evaluate_compounds_from_csv(args.csv, args.smiles_column, args.name_column, args.test)
        
        if results_df is not None:
            # Display overall statistics
            print(f"\nProcessed compounds: {len(results_df)}")
            
            # Sort by final score (if available)
            if 'final_score' in results_df.columns and not results_df['final_score'].isna().all():
                results_df = results_df.sort_values('final_score', ascending=False)
                print(f"Best score: {results_df['final_score'].max():.2f}")
                print(f"Worst score: {results_df['final_score'].min():.2f}")
                print(f"Average score: {results_df['final_score'].mean():.2f}")
            
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

if __name__ == "__main__":
    main()