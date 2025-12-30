"""
Example Usage of ATLAS Anomaly Detection System
Demonstrates all major components
"""

# ============================================================================
# Example 1: Using the Notebook (Recommended for Interactive Analysis)
# ============================================================================
"""
The atlas_anomaly_analysis.ipynb notebook provides a complete interactive
workflow. Simply run it cell-by-cell in Jupyter:

jupyter notebook atlas_anomaly_analysis.ipynb

The notebook will:
1. Load ATLAS data (ROOT files or synthetic)
2. Train ensemble anomaly detector
3. Generate predictions
4. Create visualizations
5. Export results to CSV/JSON
"""

# ============================================================================
# Example 2: Direct Python Usage
# ============================================================================

import numpy as np
import pandas as pd
from atlas_anomaly_detection import EnsembleAnomalyDetector
from sklearn.model_selection import train_test_split

# Create or load data
n_samples = 10000
np.random.seed(42)
data = pd.DataFrame({
    'lep1_pt': np.random.gamma(2, 20, n_samples) + 20,
    'lep2_pt': np.random.gamma(2, 15, n_samples) + 15,
    'lep1_eta': np.random.normal(0, 1.5, n_samples),
    'lep2_eta': np.random.normal(0, 1.5, n_samples),
    'lep_pt_ratio': np.random.lognormal(0, 0.5, n_samples),
    'lep_pt_sum': np.random.gamma(3, 25, n_samples) + 40,
    'lep_deltaR': np.random.gamma(2, 0.5, n_samples) + 0.4,
    'mll': np.random.gamma(3, 20, n_samples) + 50,
    'met_et': np.random.gamma(2, 15, n_samples) + 10,
    'jet_n': np.random.poisson(2.5, n_samples),
    'jet_pt_lead': np.random.gamma(2, 25, n_samples) + 25,
    'jet_ht': np.random.gamma(3, 30, n_samples) + 50,
    'total_pt': np.random.gamma(4, 40, n_samples) + 100,
    'centrality': np.random.beta(2, 2, n_samples),
})

# Split data
X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)

# Train ensemble
print("Training ensemble anomaly detector...")
ensemble = EnsembleAnomalyDetector(contamination=0.05)
ensemble.fit(X_train.values)

# Make predictions
print("Running anomaly detection...")
predictions = ensemble.predict(X_test.values, voting='soft')
individual_preds = ensemble.get_individual_predictions(X_test.values)

# Results
n_anomalies = np.sum(predictions == -1)
print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(predictions)*100:.2f}%)")

# ============================================================================
# Example 3: Real-Time Streaming Detection
# ============================================================================

from real_time_detector import StreamingAnomalyDetector

# Initialize streaming detector
stream_detector = StreamingAnomalyDetector(ensemble, buffer_size=1000)

# Process events one at a time (simulating real detector trigger)
print("\nProcessing events in real-time...")
for i in range(100):
    event = X_test.iloc[i].values
    result = stream_detector.process_event(event)
    
    if result['is_anomaly']:
        print(f"Event {result['event_id']}: ANOMALY detected!")
        print(f"  Score: {result['anomaly_score']:.4f}")
        print(f"  Models agreeing: {result['model_agreement']}/4")

# Get statistics
stats = stream_detector.get_statistics()
print(f"\nStreaming Statistics:")
print(f"  Total events: {stats['total_events_processed']}")
print(f"  Anomalies: {stats['total_anomalies_detected']}")
print(f"  Anomaly rate: {stats['anomaly_rate_percent']:.2f}%")
print(f"  Avg latency: {stats['avg_processing_time_ms']:.3f} ms/event")
print(f"  Throughput: {stats['throughput_events_per_sec']:.0f} events/sec")

# ============================================================================
# Example 4: Batch Processing
# ============================================================================

print("\nBatch processing...")
batch_events = X_test.iloc[100:150].values
batch_results = stream_detector.process_batch(batch_events)

anomalies_in_batch = [r for r in batch_results if r['is_anomaly']]
print(f"Processed batch of {len(batch_results)} events")
print(f"Found {len(anomalies_in_batch)} anomalies")

# ============================================================================
# Example 5: Adaptive Streaming (with threshold adaptation)
# ============================================================================

from real_time_detector import AdaptiveAnomalyDetector

print("\nAdaptive streaming detection...")
adaptive_detector = AdaptiveAnomalyDetector(ensemble, buffer_size=1000, 
                                           adaptation_window=100)

# Process more events with adaptive thresholding
for i in range(200, 300):
    event = X_test.iloc[i].values
    result = adaptive_detector.process_event(event)

adaptive_stats = adaptive_detector.get_statistics()
print(f"Adaptive Statistics:")
print(f"  Events processed: {adaptive_stats['total_events_processed']}")
print(f"  Adaptive threshold: {adaptive_stats.get('adaptive_threshold', 'N/A')}")

# ============================================================================
# Example 6: Performance Benchmarking
# ============================================================================

import time
from evaluation_metrics import PerformanceBenchmark

print("\nBenchmarking individual models...")
benchmark = PerformanceBenchmark()

# Benchmark each model
for model_name, model in ensemble.models.items():
    times = []
    for _ in range(10):
        start = time.time()
        _ = model.predict(X_test.values[:100])  # 100 events
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    throughput = 100 / avg_time
    print(f"{model_name}: {throughput:.0f} events/sec")

# ============================================================================
# Example 7: Individual Model Analysis
# ============================================================================

print("\nIndividual Model Results:")
print("-" * 50)

for model_name, predictions_model in individual_preds.items():
    n_anom = np.sum(predictions_model == -1)
    pct = n_anom / len(predictions_model) * 100
    print(f"{model_name:20s}: {n_anom:4d} anomalies ({pct:5.2f}%)")

# ============================================================================
# Example 8: Top Anomalies Analysis
# ============================================================================

print("\nTop 10 Most Anomalous Events:")
print("-" * 70)

# Calculate model agreement
anomaly_counts = np.sum([p == -1 for p in individual_preds.values()], axis=0)
top_indices = np.argsort(anomaly_counts)[-10:][::-1]

for rank, idx in enumerate(top_indices, 1):
    event = X_test.iloc[idx]
    n_models = anomaly_counts[idx]
    print(f"{rank:2d}. Event {idx:4d} | Models: {int(n_models)}/4 | "
          f"mll: {event['mll']:6.1f} GeV | MET: {event['met_et']:6.1f} GeV")

# ============================================================================
# Example 9: Anomaly Scores and Thresholding
# ============================================================================

print("\nAnomalous Reconstruction Errors (Autoencoder):")
print("-" * 50)

scores = ensemble.models['autoencoder'].anomaly_scores(X_test.values)
threshold = ensemble.models['autoencoder'].threshold

print(f"Min score: {scores.min():.4f}")
print(f"Max score: {scores.max():.4f}")
print(f"Mean score: {scores.mean():.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"Events above threshold: {np.sum(scores > threshold)}")

# ============================================================================
# Example 10: Export Results
# ============================================================================

print("\nExporting results...")
from pathlib import Path

output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

# Save predictions
results_df = X_test.copy()
results_df['prediction'] = predictions
results_df['anomaly_score'] = scores
results_df['n_models_agree'] = anomaly_counts

for model_name, model_preds in individual_preds.items():
    results_df[f'pred_{model_name}'] = model_preds

results_df.to_csv(output_dir / 'anomaly_predictions.csv', index=False)
print(f"[OK] Saved predictions to {output_dir / 'anomaly_predictions.csv'}")

# Save top anomalies
top_anomalies = results_df.iloc[top_indices]
top_anomalies.to_csv(output_dir / 'top_anomalies.csv', index=False)
print(f"[OK] Saved top anomalies to {output_dir / 'top_anomalies.csv'}")

print("\n" + "=" * 70)
print("[SUCCESS] All examples completed successfully!")
print("=" * 70)
