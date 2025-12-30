"""
Real-Time Anomaly Detection for ATLAS Particle Detector Data
Author: CERN openlab Project Candidate
Date: 2024

This system implements multiple anomaly detection algorithms for identifying
rare events in ATLAS 13 TeV collision data with exactly two leptons.
PyTorch implementation for deep learning models.
"""

import numpy as np
import pandas as pd
import uproot
import awkward as ak
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ATLASDataLoader:
    """
    Handles loading and preprocessing of ATLAS Open Data ROOT files
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.features = None
        self.data = None
        
    def load_root_file(self, filename: str, max_events: Optional[int] = None) -> pd.DataFrame:
        """
        Load ROOT file and extract physics features
        
        Args:
            filename: Path to ROOT file
            max_events: Maximum number of events to load (None = all)
            
        Returns:
            DataFrame with physics features
        """
        try:
            file = uproot.open(filename)
            tree = file["mini"]  # ATLAS Open Data uses 'mini' tree
            
            # Key physics variables for 2-lepton events
            branches = [
                # Lepton features
                "lep_pt", "lep_eta", "lep_phi", "lep_E",
                "lep_charge", "lep_type",
                # Missing transverse energy
                "met_et", "met_phi",
                # Jets
                "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_m",
                # Event-level
                "mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE",
                "scaleFactor_MUON", "scaleFactor_BTAG", "scaleFactor_LepTRIGGER"
            ]
            
            # Load data
            data = tree.arrays(branches, library="pd", entry_stop=max_events)
            
            return data
            
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            return None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer high-level physics features from raw data
        
        Args:
            df: Raw DataFrame from ROOT file
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Lepton kinematics (assuming 2 leptons per event)
        if isinstance(df['lep_pt'].iloc[0], (list, np.ndarray)):
            # Handle array columns
            features['lep1_pt'] = df['lep_pt'].apply(lambda x: x[0] if len(x) > 0 else 0)
            features['lep2_pt'] = df['lep_pt'].apply(lambda x: x[1] if len(x) > 1 else 0)
            features['lep1_eta'] = df['lep_eta'].apply(lambda x: x[0] if len(x) > 0 else 0)
            features['lep2_eta'] = df['lep_eta'].apply(lambda x: x[1] if len(x) > 1 else 0)
            features['lep1_phi'] = df['lep_phi'].apply(lambda x: x[0] if len(x) > 0 else 0)
            features['lep2_phi'] = df['lep_phi'].apply(lambda x: x[1] if len(x) > 1 else 0)
            features['lep1_E'] = df['lep_E'].apply(lambda x: x[0] if len(x) > 0 else 0)
            features['lep2_E'] = df['lep_E'].apply(lambda x: x[1] if len(x) > 1 else 0)
        else:
            # Handle scalar columns (already processed)
            features['lep1_pt'] = df['lep_pt']
            features['lep2_pt'] = df.get('lep_pt', 0)
        
        # Derived features
        features['lep_pt_ratio'] = features['lep1_pt'] / (features['lep2_pt'] + 1e-6)
        features['lep_pt_sum'] = features['lep1_pt'] + features['lep2_pt']
        features['lep_pt_diff'] = np.abs(features['lep1_pt'] - features['lep2_pt'])
        
        # Delta R between leptons
        delta_eta = features['lep1_eta'] - features['lep2_eta']
        delta_phi = features['lep1_phi'] - features['lep2_phi']
        # Wrap phi to [-pi, pi]
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
        features['lep_deltaR'] = np.sqrt(delta_eta**2 + delta_phi**2)
        
        # Invariant mass of dilepton system
        features['mll'] = np.sqrt(
            (features['lep1_E'] + features['lep2_E'])**2 -
            (features['lep1_pt'] + features['lep2_pt'])**2
        )
        
        # Missing ET
        features['met_et'] = df['met_et']
        features['met_phi'] = df['met_phi']
        
        # Jet features
        features['jet_n'] = df['jet_n']
        if isinstance(df['jet_pt'].iloc[0], (list, np.ndarray)):
            features['jet_pt_lead'] = df['jet_pt'].apply(lambda x: x[0] if len(x) > 0 else 0)
            features['jet_pt_sum'] = df['jet_pt'].apply(lambda x: np.sum(x) if len(x) > 0 else 0)
            features['jet_ht'] = features['jet_pt_sum']  # HT = scalar sum of jet pT
        else:
            features['jet_pt_lead'] = df.get('jet_pt', 0)
            features['jet_pt_sum'] = df.get('jet_pt', 0)
            features['jet_ht'] = features['jet_pt_sum']
        
        # Total transverse momentum
        features['total_pt'] = features['lep_pt_sum'] + features['jet_pt_sum'] + features['met_et']
        
        # Centrality
        features['centrality'] = (features['lep_pt_sum'] + features['met_et']) / (features['total_pt'] + 1e-6)
        
        # Event weights (for MC)
        if 'mcWeight' in df.columns:
            features['weight'] = df['mcWeight']
        else:
            features['weight'] = 1.0
        
        return features
    
    def prepare_data(self, filenames: List[str], max_events_per_file: Optional[int] = None) -> pd.DataFrame:
        """
        Load and prepare data from multiple ROOT files
        
        Args:
            filenames: List of ROOT file paths
            max_events_per_file: Maximum events per file
            
        Returns:
            Combined and processed DataFrame
        """
        all_data = []
        
        for filename in filenames:
            print(f"Loading {filename}...")
            df = self.load_root_file(filename, max_events_per_file)
            if df is not None:
                features = self.engineer_features(df)
                all_data.append(features)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(combined)} events with {len(combined.columns)} features")
            return combined
        else:
            return pd.DataFrame()


class PyTorchAutoencoder(nn.Module):
    """
    PyTorch Autoencoder architecture for anomaly detection
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(PyTorchAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class AutoencoderAnomalyDetector:
    """
    Deep Autoencoder for unsupervised anomaly detection using PyTorch
    """
    
    def __init__(self, encoding_dim: int = 8, contamination: float = 0.05, device: str = 'cpu'):
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.device = torch.device(device)
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.optimizer = None
        self.training_losses = []
        self.validation_losses = []
        
    def build_model(self, input_dim: int):
        """
        Build autoencoder model
        
        Args:
            input_dim: Number of input features
        """
        self.model = PyTorchAutoencoder(input_dim, self.encoding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = TorchReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    
    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 256, validation_split: float = 0.2):
        """
        Train the autoencoder
        
        Args:
            X: Training data (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.build_model(X.shape[1])
        
        # Split into train/validation
        n_train = int(len(X_scaled) * (1 - validation_split))
        indices = np.random.permutation(len(X_scaled))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X_scaled[train_indices]
        X_val = X_scaled[val_indices]
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.model.train()
            
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                
                self.optimizer.zero_grad()
                reconstructed = self.model(X_batch)
                loss = criterion(reconstructed, X_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(X_train)
            self.training_losses.append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            self.model.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch[0].to(self.device)
                    reconstructed = self.model(X_batch)
                    loss = criterion(reconstructed, X_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(X_val)
            self.validation_losses.append(val_loss)
            
            # Early stopping and learning rate scheduling
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Calculate reconstruction errors on training data for threshold
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructions = self.model(X_train_tensor).cpu().numpy()
        
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(mse, 100 * (1 - self.contamination))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal)
        
        Args:
            X: Data to predict
            
        Returns:
            Array of predictions
        """
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructions = self.model(X_tensor).cpu().numpy()
        
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        predictions = np.where(mse > self.threshold, -1, 1)
        return predictions
    
    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (reconstruction error)
        
        Args:
            X: Data to score
            
        Returns:
            Array of anomaly scores
        """
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructions = self.model(X_tensor).cpu().numpy()
        
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        return mse


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection algorithms
    """
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = RobustScaler()
        
        # Initialize models
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                n_estimators=200,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=contamination,
                random_state=42
            ),
            'autoencoder': AutoencoderAnomalyDetector(
                encoding_dim=8,
                contamination=contamination
            )
        }
        
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit all models in the ensemble
        
        Args:
            X: Training data
        """
        print("Fitting ensemble models...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit classical ML models
        print("  - Isolation Forest...")
        self.models['isolation_forest'].fit(X_scaled)
        
        print("  - One-Class SVM...")
        self.models['one_class_svm'].fit(X_scaled)
        
        print("  - Elliptic Envelope...")
        self.models['elliptic_envelope'].fit(X_scaled)
        
        # Fit deep learning model
        print("  - Autoencoder...")
        self.models['autoencoder'].fit(X, epochs=50, batch_size=256)
        
        self.fitted = True
        print("Ensemble training complete!")
    
    def predict(self, X: np.ndarray, voting: str = 'soft') -> np.ndarray:
        """
        Predict using ensemble voting
        
        Args:
            X: Data to predict
            voting: 'soft' (majority vote) or 'hard' (unanimous)
            
        Returns:
            Array of predictions
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        predictions['isolation_forest'] = self.models['isolation_forest'].predict(X_scaled)
        predictions['one_class_svm'] = self.models['one_class_svm'].predict(X_scaled)
        predictions['elliptic_envelope'] = self.models['elliptic_envelope'].predict(X_scaled)
        predictions['autoencoder'] = self.models['autoencoder'].predict(X)
        
        # Combine predictions
        pred_array = np.array(list(predictions.values()))
        
        if voting == 'soft':
            # Majority vote (at least 2 models agree it's anomaly)
            anomaly_votes = np.sum(pred_array == -1, axis=0)
            final_predictions = np.where(anomaly_votes >= 2, -1, 1)
        else:  # hard
            # All models must agree
            final_predictions = np.where(np.all(pred_array == -1, axis=0), -1, 1)
        
        return final_predictions
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Args:
            X: Data to predict
            
        Returns:
            Dictionary of predictions per model
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        predictions['isolation_forest'] = self.models['isolation_forest'].predict(X_scaled)
        predictions['one_class_svm'] = self.models['one_class_svm'].predict(X_scaled)
        predictions['elliptic_envelope'] = self.models['elliptic_envelope'].predict(X_scaled)
        predictions['autoencoder'] = self.models['autoencoder'].predict(X)
        
        return predictions


class AnomalyVisualizer:
    """
    Visualization utilities for anomaly detection results
    """
    
    @staticmethod
    def plot_training_history(detector: 'AutoencoderAnomalyDetector'):
        """Plot autoencoder training history"""
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        
        axes.plot(detector.training_losses, label='Training Loss', marker='o', markersize=3)
        axes.plot(detector.validation_losses, label='Validation Loss', marker='s', markersize=3)
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss (MSE)')
        axes.set_title('Autoencoder Training History - PyTorch')
        axes.legend()
        axes.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_distributions(data: pd.DataFrame, predictions: np.ndarray, features: List[str]):
        """Plot feature distributions for normal vs anomalous events"""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Plot distributions
            ax.hist(data[feature][normal_mask], bins=50, alpha=0.6, label='Normal', color='blue', density=True)
            ax.hist(data[feature][anomaly_mask], bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_anomaly_scores(scores: np.ndarray, threshold: float, title: str = 'Anomaly Scores'):
        """Plot anomaly score distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(scores, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax.set_xlabel('Anomaly Score (Reconstruction Error)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pca_visualization(data: pd.DataFrame, predictions: np.ndarray):
        """Visualize anomalies in 2D PCA space"""
        # Perform PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        ax.scatter(data_pca[normal_mask, 0], data_pca[normal_mask, 1], 
                  c='blue', alpha=0.3, s=20, label='Normal')
        ax.scatter(data_pca[anomaly_mask, 0], data_pca[anomaly_mask, 1], 
                  c='red', alpha=0.7, s=50, label='Anomaly', marker='^')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Anomaly Detection in PCA Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_model_agreement(individual_predictions: Dict[str, np.ndarray]):
        """Visualize agreement between different models"""
        models = list(individual_predictions.keys())
        n_samples = len(individual_predictions[models[0]])
        
        # Calculate anomaly counts per model
        anomaly_counts = {model: np.sum(pred == -1) for model, pred in individual_predictions.items()}
        
        # Calculate agreement matrix
        agreement_matrix = np.zeros((len(models), len(models)))
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i <= j:
                    agreement = np.sum(
                        individual_predictions[model1] == individual_predictions[model2]
                    ) / n_samples
                    agreement_matrix[i, j] = agreement
                    agreement_matrix[j, i] = agreement
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Anomaly counts per model
        axes[0].bar(range(len(models)), anomaly_counts.values(), color='steelblue')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
        axes[0].set_ylabel('Number of Anomalies Detected')
        axes[0].set_title('Anomalies Detected by Each Model')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Agreement heatmap
        im = axes[1].imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_xticks(range(len(models)))
        axes[1].set_yticks(range(len(models)))
        axes[1].set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
        axes[1].set_yticklabels([m.replace('_', ' ').title() for m in models])
        axes[1].set_title('Model Agreement Matrix')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(models)):
                text = axes[1].text(j, i, f'{agreement_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=axes[1], label='Agreement Score')
        plt.tight_layout()
        return fig


def main():
    """
    Main execution pipeline
    """
    print("=" * 80)
    print("ATLAS Particle Detector - Real-Time Anomaly Detection System")
    print("=" * 80)
    
    # Example usage with synthetic data (replace with actual ROOT files)
    print("\n[INFO] Generating synthetic ATLAS-like data for demonstration...")
    print("       (Replace this with actual ROOT file loading in production)")
    
    # Simulate ATLAS features
    n_samples = 10000
    np.random.seed(42)
    
    # Create realistic feature distributions
    data = pd.DataFrame({
        'lep1_pt': np.random.gamma(2, 20, n_samples) + 20,  # Leading lepton pT
        'lep2_pt': np.random.gamma(2, 15, n_samples) + 15,  # Sub-leading lepton pT
        'lep1_eta': np.random.normal(0, 1.5, n_samples),
        'lep2_eta': np.random.normal(0, 1.5, n_samples),
        'lep_pt_ratio': np.random.lognormal(0, 0.5, n_samples),
        'lep_pt_sum': np.random.gamma(3, 25, n_samples) + 40,
        'lep_deltaR': np.random.gamma(2, 0.5, n_samples) + 0.4,
        'mll': np.random.gamma(3, 20, n_samples) + 50,  # Invariant mass
        'met_et': np.random.gamma(2, 15, n_samples) + 10,  # Missing ET
        'jet_n': np.random.poisson(2.5, n_samples),
        'jet_pt_lead': np.random.gamma(2, 25, n_samples) + 25,
        'jet_ht': np.random.gamma(3, 30, n_samples) + 50,
        'total_pt': np.random.gamma(4, 40, n_samples) + 100,
        'centrality': np.random.beta(2, 2, n_samples),
    })
    
    # Inject some anomalies (simulate BSM signals or rare processes)
    n_anomalies = 200
    anomaly_mask = np.zeros(n_samples, dtype=bool)
    anomaly_mask[np.random.choice(n_samples, n_anomalies, replace=False)] = True
    
    # Make anomalies have extreme feature values
    data.loc[anomaly_mask, 'mll'] = data.loc[anomaly_mask, 'mll'] * 3
    data.loc[anomaly_mask, 'met_et'] = data.loc[anomaly_mask, 'met_et'] * 2
    data.loc[anomaly_mask, 'lep_pt_sum'] = data.loc[anomaly_mask, 'lep_pt_sum'] * 2
    
    print(f"[INFO] Created dataset with {n_samples} events")
    print(f"[INFO] Injected {n_anomalies} synthetic anomalies")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)
    
    print(f"[INFO] Training set: {len(X_train)} events")
    print(f"[INFO] Test set: {len(X_test)} events")
    
    # Initialize and train ensemble
    print("\n" + "=" * 80)
    print("Training Ensemble Anomaly Detector")
    print("=" * 80)
    
    ensemble = EnsembleAnomalyDetector(contamination=0.05)
    ensemble.fit(X_train.values)
    
    # Predict on test set
    print("\n[INFO] Running anomaly detection on test set...")
    predictions = ensemble.predict(X_test.values, voting='soft')
    individual_preds = ensemble.get_individual_predictions(X_test.values)
    
    n_anomalies_detected = np.sum(predictions == -1)
    anomaly_rate = n_anomalies_detected / len(predictions) * 100
    
    print(f"[RESULT] Detected {n_anomalies_detected} anomalies ({anomaly_rate:.2f}%)")
    
    # Show individual model results
    print("\n[INFO] Individual model results:")
    for model_name, preds in individual_preds.items():
        n_anom = np.sum(preds == -1)
        print(f"  - {model_name}: {n_anom} anomalies ({n_anom/len(preds)*100:.2f}%)")
    
    # Visualizations
    print("\n[INFO] Generating visualizations...")
    
    # Feature distributions
    key_features = ['mll', 'met_et', 'lep_pt_sum', 'lep_deltaR', 'jet_ht', 'centrality']
    fig1 = AnomalyVisualizer.plot_feature_distributions(X_test, predictions, key_features)
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: feature_distributions.png")
    
    # PCA visualization
    fig2 = AnomalyVisualizer.plot_pca_visualization(X_test, predictions)
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: pca_visualization.png")
    
    # Model agreement
    fig3 = AnomalyVisualizer.plot_model_agreement(individual_preds)
    plt.savefig('model_agreement.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: model_agreement.png")
    
    # Anomaly scores (for autoencoder)
    autoencoder_scores = ensemble.models['autoencoder'].anomaly_scores(X_test.values)
    fig4 = AnomalyVisualizer.plot_anomaly_scores(
        autoencoder_scores, 
        ensemble.models['autoencoder'].threshold,
        'Autoencoder Reconstruction Error'
    )
    plt.savefig('anomaly_scores.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: anomaly_scores.png")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total events analyzed: {len(X_test)}")
    print(f"Anomalies detected: {n_anomalies_detected} ({anomaly_rate:.2f}%)")
    print(f"\nTop anomalous events (by model consensus):")
    
    # Find events flagged by multiple models
    anomaly_counts = np.sum([p == -1 for p in individual_preds.values()], axis=0)
    top_anomaly_indices = np.argsort(anomaly_counts)[-10:][::-1]
    
    for rank, idx in enumerate(top_anomaly_indices, 1):
        n_models = anomaly_counts[idx]
        print(f"  {rank}. Event {idx}: Flagged by {n_models}/4 models")
        print(f"     mll={X_test.iloc[idx]['mll']:.1f} GeV, "
              f"MET={X_test.iloc[idx]['met_et']:.1f} GeV, "
              f"ΣpT(lep)={X_test.iloc[idx]['lep_pt_sum']:.1f} GeV")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check output PNG files for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()