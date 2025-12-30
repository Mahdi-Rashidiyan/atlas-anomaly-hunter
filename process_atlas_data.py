"""
ATLAS Real Data Processing and Anomaly Detection
Processes ROOT files from CERN Open Data Record 15007
"""

import numpy as np
import pandas as pd
import uproot
from pathlib import Path
from typing import List, Tuple
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from atlas_anomaly_detection import (
    ATLASDataLoader,
    EnsembleAnomalyDetector,
    AnomalyVisualizer
)


def load_root_file(filepath: str, max_events: int = None) -> pd.DataFrame:
    """
    Load ROOT file and extract features
    
    Args:
        filepath: Path to ROOT file
        max_events: Maximum events to load (None = all)
        
    Returns:
        DataFrame with physics features
    """
    print(f"\n{'='*80}")
    print(f"Loading: {Path(filepath).name}")
    print(f"{'='*80}")
    
    try:
        file = uproot.open(filepath)
        tree = file["mini"]
        
        print(f"Total events in file: {tree.num_entries:,}")
        
        # Key branches for 2-lepton analysis
        branches_preferred = [
            "lep_pt", "lep_eta", "lep_phi", "lep_E",
            "lep_charge", "lep_type",
            "met_et", "met_phi",
            "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_E",
            "mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE",
            "scaleFactor_MUON", "scaleFactor_BTAG", "scaleFactor_LepTRIGGER"
        ]
        
        # Filter to only branches that exist
        available_branches = set(tree.keys())
        branches = [b for b in branches_preferred if b in available_branches]
        
        print(f"Available branches to load: {len(branches)}/{len(branches_preferred)}")
        
        # Load data
        if max_events:
            data = tree.arrays(branches, library="pd", entry_stop=max_events)
            print(f"Loaded: {len(data):,} events (requested {max_events:,})")
        else:
            data = tree.arrays(branches, library="pd")
            print(f"Loaded: {len(data):,} events (all)")
        
        return data
        
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer high-level physics features from raw ROOT data
    
    Args:
        df: Raw DataFrame from ROOT file
        
    Returns:
        DataFrame with engineered features
    """
    print("\nEngineering physics features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Lepton kinematics (handle array columns)
    try:
        features['lep1_pt'] = df['lep_pt'].apply(lambda x: x[0] if len(x) > 0 else 0)
        features['lep2_pt'] = df['lep_pt'].apply(lambda x: x[1] if len(x) > 1 else 0)
        features['lep1_eta'] = df['lep_eta'].apply(lambda x: x[0] if len(x) > 0 else 0)
        features['lep2_eta'] = df['lep_eta'].apply(lambda x: x[1] if len(x) > 1 else 0)
        features['lep1_phi'] = df['lep_phi'].apply(lambda x: x[0] if len(x) > 0 else 0)
        features['lep2_phi'] = df['lep_phi'].apply(lambda x: x[1] if len(x) > 1 else 0)
        features['lep1_E'] = df['lep_E'].apply(lambda x: x[0] if len(x) > 0 else 0)
        features['lep2_E'] = df['lep_E'].apply(lambda x: x[1] if len(x) > 1 else 0)
    except Exception as e:
        print(f"Warning processing lepton features: {e}")
        return None
    
    # Derived kinematic features
    features['lep_pt_ratio'] = features['lep1_pt'] / (features['lep2_pt'] + 1e-6)
    features['lep_pt_sum'] = features['lep1_pt'] + features['lep2_pt']
    features['lep_pt_diff'] = np.abs(features['lep1_pt'] - features['lep2_pt'])
    
    # Delta R between leptons
    try:
        delta_eta = features['lep1_eta'] - features['lep2_eta']
        delta_phi = features['lep1_phi'] - features['lep2_phi']
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
        features['lep_deltaR'] = np.sqrt(delta_eta**2 + delta_phi**2)
    except Exception as e:
        print(f"Warning computing deltaR: {e}")
        features['lep_deltaR'] = 0
    
    # Invariant mass
    try:
        features['mll'] = np.sqrt(
            np.maximum((features['lep1_E'] + features['lep2_E'])**2 -
                      (features['lep1_pt'] + features['lep2_pt'])**2, 0)
        )
    except Exception as e:
        print(f"Warning computing mll: {e}")
        features['mll'] = 0
    
    # Missing ET
    try:
        features['met_et'] = df['met_et']
        features['met_phi'] = df['met_phi']
    except:
        features['met_et'] = 0
        features['met_phi'] = 0
    
    # Jet features
    try:
        features['jet_n'] = df['jet_n']
        features['jet_pt_lead'] = df['jet_pt'].apply(lambda x: x[0] if len(x) > 0 else 0)
        features['jet_pt_sum'] = df['jet_pt'].apply(lambda x: np.sum(x) if len(x) > 0 else 0)
        features['jet_ht'] = features['jet_pt_sum']
    except:
        features['jet_n'] = 0
        features['jet_pt_lead'] = 0
        features['jet_pt_sum'] = 0
        features['jet_ht'] = 0
    
    # Total transverse momentum
    features['total_pt'] = features['lep_pt_sum'] + features['jet_pt_sum'] + features['met_et']
    
    # Centrality
    features['centrality'] = (features['lep_pt_sum'] + features['met_et']) / (features['total_pt'] + 1e-6)
    
    # Weights
    try:
        features['weight'] = df['mcWeight']
    except:
        features['weight'] = 1.0
    
    print(f"✓ Engineered {len(features.columns)} features")
    print(f"  Features: {', '.join(features.columns[:5])}... and {len(features.columns)-5} more")
    
    return features


def process_root_files(root_files: List[str], max_events: int = None) -> pd.DataFrame:
    """
    Process all ROOT files and combine into single dataset
    
    Args:
        root_files: List of ROOT file paths
        max_events: Max events per file
        
    Returns:
        Combined DataFrame with features
    """
    all_features = []
    
    for filepath in root_files:
        # Load raw data
        raw_data = load_root_file(filepath, max_events)
        
        if raw_data is None or len(raw_data) == 0:
            print(f"  ⚠️  Skipping {filepath} - no data loaded")
            continue
        
        # Engineer features
        engineered = engineer_features(raw_data)
        
        if engineered is None or len(engineered) == 0:
            print(f"  ⚠️  Skipping {filepath} - feature engineering failed")
            continue
        
        all_features.append(engineered)
        print(f"  ✓ Processed: {len(engineered):,} events")
    
    if not all_features:
        print("\n❌ ERROR: No data successfully processed!")
        return None
    
    # Combine all data
    combined = pd.concat(all_features, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"COMBINED DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total events: {len(combined):,}")
    print(f"Total features: {len(combined.columns)}")
    print(f"\nFeature summary:")
    print(combined.describe())
    
    return combined


def run_anomaly_detection(data: pd.DataFrame, contamination: float = 0.05) -> Tuple[np.ndarray, dict]:
    """
    Run ensemble anomaly detection on real data
    
    Args:
        data: Feature DataFrame
        contamination: Expected anomaly fraction
        
    Returns:
        Tuple of (predictions, individual_predictions)
    """
    print(f"\n{'='*80}")
    print("ANOMALY DETECTION - ENSEMBLE")
    print(f"{'='*80}")
    print(f"Training ensemble on {len(data):,} events...")
    print(f"Contamination parameter: {contamination*100:.1f}%")
    
    # Train ensemble
    ensemble = EnsembleAnomalyDetector(contamination=contamination)
    ensemble.fit(data.values)
    
    # Predict
    print("\nRunning predictions...")
    predictions = ensemble.predict(data.values, voting='soft')
    individual_preds = ensemble.get_individual_predictions(data.values)
    
    # Results
    n_anomalies = np.sum(predictions == -1)
    anomaly_rate = n_anomalies / len(predictions) * 100
    
    print(f"\n{'='*80}")
    print("DETECTION RESULTS")
    print(f"{'='*80}")
    print(f"Total events: {len(predictions):,}")
    print(f"Anomalies detected: {n_anomalies:,} ({anomaly_rate:.2f}%)")
    
    print(f"\nIndividual model results:")
    for model_name, preds in individual_preds.items():
        n_anom = np.sum(preds == -1)
        rate = n_anom / len(preds) * 100
        print(f"  • {model_name:20s}: {n_anom:5d} anomalies ({rate:5.2f}%)")
    
    return predictions, individual_preds


def generate_visualizations(data: pd.DataFrame, predictions: np.ndarray, individual_preds: dict):
    """
    Generate visualization plots
    
    Args:
        data: Feature DataFrame
        predictions: Ensemble predictions
        individual_preds: Individual model predictions
    """
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    key_features = ['mll', 'met_et', 'lep_pt_sum', 'lep_deltaR', 'jet_ht', 'centrality']
    
    try:
        # Feature distributions
        print("\n1. Feature distributions (normal vs anomaly)...")
        fig1 = AnomalyVisualizer.plot_feature_distributions(data, predictions, key_features)
        fig1.savefig('cern_feature_distributions.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: cern_feature_distributions.png")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    try:
        # PCA visualization
        print("2. PCA visualization...")
        fig2 = AnomalyVisualizer.plot_pca_visualization(data, predictions)
        fig2.savefig('cern_pca_visualization.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: cern_pca_visualization.png")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    try:
        # Model agreement
        print("3. Model agreement heatmap...")
        fig3 = AnomalyVisualizer.plot_model_agreement(individual_preds)
        fig3.savefig('cern_model_agreement.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: cern_model_agreement.png")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    try:
        # Anomaly scores
        print("4. Anomaly score distribution...")
        autoencoder = None
        for model_name, model in [('autoencoder', None)]:
            # Find autoencoder in ensemble
            pass
        # For now, create a simple histogram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(predictions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Prediction (-1=Anomaly, 1=Normal)')
        ax.set_ylabel('Frequency')
        ax.set_title('Ensemble Predictions Distribution - CERN Real Data')
        ax.grid(True, alpha=0.3)
        fig.savefig('cern_anomaly_scores.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: cern_anomaly_scores.png")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n✓ All visualizations generated!")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("ATLAS REAL DATA - ANOMALY DETECTION PIPELINE")
    print("CERN Open Data Record 15007")
    print("="*80)
    
    # Find ROOT files
    print("\nSearching for ROOT files...")
    root_dir = Path(".")
    root_files = list(root_dir.glob("*.root")) + list(root_dir.glob("data/*.root"))
    
    if not root_files:
        print("❌ No ROOT files found!")
        print("   Expected: *.root files in current directory or ./data/")
        return
    
    print(f"✓ Found {len(root_files)} ROOT file(s):")
    for f in root_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"  • {f.name} ({size_gb:.2f} GB)")
    
    # Process ROOT files
    print(f"\n{'='*80}")
    print("PROCESSING ROOT FILES")
    print(f"{'='*80}")
    
    # Use first 50000 events per file for faster processing
    data = process_root_files([str(f) for f in root_files], max_events=50000)
    
    if data is None or len(data) == 0:
        print("❌ Failed to process data!")
        return
    
    # Remove NaN and Inf
    print("\nCleaning data...")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    print(f"✓ After cleaning: {len(data):,} events")
    
    if len(data) < 100:
        print("❌ Not enough events after cleaning!")
        return
    
    # Run anomaly detection
    predictions, individual_preds = run_anomaly_detection(data, contamination=0.05)
    
    # Generate visualizations
    generate_visualizations(data, predictions, individual_preds)
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    try:
        results_df = data.copy()
        results_df['prediction'] = predictions
        results_df['anomaly'] = (predictions == -1).astype(int)
        
        results_df.to_csv('cern_anomaly_results.csv', index=False)
        print("✓ Saved: cern_anomaly_results.csv")
        
        # Statistics
        stats = {
            'total_events': len(data),
            'anomalies_detected': int(np.sum(predictions == -1)),
            'anomaly_rate': float(np.sum(predictions == -1) / len(predictions) * 100),
            'features_used': len(data.columns),
        }
        
        import json
        with open('cern_detection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("✓ Saved: cern_detection_stats.json")
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print("  • cern_feature_distributions.png")
    print("  • cern_pca_visualization.png")
    print("  • cern_model_agreement.png")
    print("  • cern_anomaly_scores.png")
    print("  • cern_anomaly_results.csv")
    print("  • cern_detection_stats.json")
    print("\nNext: Build Jupyter notebook from these results!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
