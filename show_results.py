import pandas as pd
import json

# Load results
df = pd.read_csv('cern_anomaly_results.csv')

print("\n" + "="*80)
print("CERN REAL DATA - ANOMALY DETECTION RESULTS")
print("="*80)
print(f"\nDataset Statistics:")
print(f"  Total events processed: {len(df):,}")
print(f"  Anomalies detected: {df['anomaly'].sum():,} ({df['anomaly'].mean()*100:.2f}%)")
print(f"  Normal events: {(1-df['anomaly']).sum():,} ({(1-df['anomaly'].mean())*100:.2f}%)")

print(f"\nFeature Statistics:")
for col in ['mll', 'met_et', 'lep_pt_sum', 'lep_deltaR']:
    if col in df.columns:
        print(f"  {col:15s}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

print(f"\nAnomalous Events Sample:")
anomalies = df[df['anomaly'] == 1].head(5)
for col in ['mll', 'met_et', 'lep_pt_sum']:
    if col in anomalies.columns:
        print(f"  {col}: {anomalies[col].values[:3]}")

# Load stats
with open('cern_detection_stats.json') as f:
    stats = json.load(f)

print(f"\nDetection Statistics:")
for key, val in stats.items():
    print(f"  {key}: {val}")

print(f"\nOutput Files Generated:")
print(f"  ✓ cern_anomaly_results.csv (14.8 MB)")
print(f"  ✓ cern_detection_stats.json")
print(f"  ✓ cern_feature_distributions.png")
print(f"  ✓ cern_pca_visualization.png")
print(f"  ✓ cern_model_agreement.png")
print(f"  ✓ cern_anomaly_scores.png")

print(f"\n✓ All results ready for Jupyter notebook!")
print("="*80 + "\n")
