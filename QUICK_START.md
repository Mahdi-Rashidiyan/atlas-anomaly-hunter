# Quick Start Guide - ATLAS Anomaly Detection

## 🚀 Get Started in 2 Minutes

### Step 1: Launch Jupyter
```bash
cd C:\Users\Liver\OneDrive\Desktop\anomaly_detection
jupyter notebook atlas_anomaly_analysis.ipynb
```

### Step 2: Run All Cells
- Press `Ctrl+A` to select all cells
- Press `Shift+Enter` to execute all

**That's it!** The notebook will:
1. Load or generate ATLAS data
2. Train 4 ensemble models
3. Detect anomalies
4. Generate 8+ visualizations
5. Export results to CSV/JSON

---

## 📊 What You'll Get

### Visualizations
- Feature distributions (normal vs anomalous)
- 2D PCA projection with anomalies
- Model agreement heatmap
- Anomaly score distribution
- ROC curve
- Performance benchmarks

### Results Files
```
outputs/
├── anomaly_predictions.csv    # All 3,000 test events with scores
├── top_anomalies.csv          # Top 20 anomalies
├── benchmark_results.json     # Model timing statistics
└── summary_statistics.json    # Overall metrics
```

### Metrics Displayed
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Anomaly Rate
- Model agreement statistics
- Inference latency per event
- Throughput (events/second)

---

## 🔧 Customize Your Analysis

### Configuration Cell
```python
CONFIG = {
    'contamination': 0.05,       # Expected anomaly fraction
    'max_events': 50000,         # Max events to load
    'test_split': 0.3,           # 70% train, 30% test
    'autoencoder_epochs': 50,    # Training iterations
}
```

### Data Source
The notebook automatically:
1. **Tries** to load ATLAS ROOT files from `./data/`
2. **Falls back** to synthetic data if files not found

To use real data:
1. Place ROOT files in `./data/` directory
2. Update `ROOT_FILES` list in Configuration cell
3. Re-run the notebook

---

## 💡 Usage Patterns

### Pattern 1: Quick Demo (synthetic data)
- Just run the notebook as-is
- Generates 10,000 synthetic ATLAS-like events
- Trains and evaluates in ~30 seconds

### Pattern 2: Real ATLAS Data
- Provide ROOT files
- Point notebook to them
- Automatically processes and analyzes

### Pattern 3: Streaming Detection
See `USAGE_EXAMPLES.py` for real-time processing:
```python
from real_time_detector import StreamingAnomalyDetector

detector = StreamingAnomalyDetector(ensemble)
result = detector.process_event(event)  # ~8ms latency
```

### Pattern 4: Batch Processing
```python
batch_results = detector.process_batch(100_events)  # Fast
```

### Pattern 5: Adaptive Thresholding
```python
from real_time_detector import AdaptiveAnomalyDetector

adaptive = AdaptiveAnomalyDetector(ensemble)
# Automatically adjusts threshold as new data arrives
```

---

## 📈 Ensemble Methods Explained

The notebook uses 4 complementary algorithms:

| Model | Best At | Speed | Memory |
|-------|---------|-------|--------|
| **Isolation Forest** | Global outliers | ⚡⚡⚡ Fast | Low |
| **One-Class SVM** | Complex boundaries | ⚡⚡ Medium | Medium |
| **Elliptic Envelope** | Covariance patterns | ⚡⚡⚡ Fast | Low |
| **Autoencoder** | Non-linear patterns | ⚡⚡ Medium | Medium |

**Voting Strategy**: An event is flagged as anomaly if **2 or more models agree**

---

## 🎯 Key Physics Features

The notebook analyzes 14 ATLAS features:

**Lepton Variables** (measurements of electrons/muons)
- `lep1_pt`, `lep2_pt`: Transverse momentum
- `lep1_eta`, `lep2_eta`: Pseudorapidity
- `lep_pt_ratio`: Leading/subleading momentum ratio
- `lep_deltaR`: Angular separation

**System Variables** (properties of the event)
- `mll`: Invariant mass of dilepton system
- `met_et`: Missing transverse energy
- `lep_pt_sum`: Total lepton momentum

**Jet Variables** (measurements of quark jets)
- `jet_n`: Number of jets
- `jet_pt_lead`: Highest jet momentum
- `jet_ht`: Scalar sum of jet momenta

**Event Variables**
- `total_pt`: Total transverse momentum
- `centrality`: Event geometry metric

---

## 🔍 Interpreting Results

### Anomaly Score
- **Range**: 0 (normal) to 2+ (anomalous)
- **Source**: Autoencoder reconstruction error
- **Threshold**: Automatically set to 95th percentile

### Model Agreement
- **0/4**: No models flagged it
- **1/4**: Controversial (edge case)
- **2/4**: Likely anomaly (default threshold)
- **3-4/4**: Definite anomaly (high confidence)

### Top Anomalies
Ranked by:
1. Number of models agreeing
2. Reconstruction error magnitude
3. Feature extremeness

---

## ⚡ Performance Tips

### Faster Training
```python
CONFIG['autoencoder_epochs'] = 10  # Instead of 50
CONFIG['max_events'] = 5000        # Instead of 50000
```

### Better Accuracy
```python
CONFIG['autoencoder_epochs'] = 100
CONFIG['batch_size'] = 512
```

### Memory Efficient
```python
CONFIG['max_events'] = 1000  # Smaller dataset
# Process in batches instead
```

---

## 🐛 Troubleshooting

### Issue: Slow Training
**Solution**: Reduce `autoencoder_epochs` in Configuration

### Issue: Out of Memory
**Solution**: Reduce `max_events` or use batch processing

### Issue: ROOT File Not Found
**Solution**: Notebook automatically uses synthetic data instead

### Issue: GPU Not Available
**Solution**: PyTorch automatically falls back to CPU (slower but works)

### Issue: Import Errors
**Solution**: Make sure all `.py` files are in the same directory

---

## 📚 Files Explained

| File | Purpose |
|------|---------|
| `atlas_anomaly_analysis.ipynb` | **Interactive analysis** (run this!) |
| `atlas_anomaly_detection.py` | Core detection algorithms |
| `real_time_detector.py` | Real-time streaming detector |
| `evaluation_metrics.py` | Performance metrics & benchmarking |
| `USAGE_EXAMPLES.py` | 10 code examples (copy & adapt) |
| `COMPLETION_REPORT.md` | Detailed technical report |
| `SETUP_COMPLETE.md` | Full setup documentation |

---

## 🎓 Learning Path

1. **Start**: Run notebook with default settings (2 min)
2. **Explore**: Change contamination rate, see results (5 min)
3. **Learn**: Read top anomalies details (10 min)
4. **Customize**: Load your own data (varies)
5. **Deploy**: Use StreamingAnomalyDetector for real-time (varies)

---

## 💾 Output Examples

### anomaly_predictions.csv
```
lep1_pt,lep2_pt,mll,met_et,...,prediction,anomaly_score,n_models_agree
45.2,32.1,75.3,12.4,...,1,0.234,0
52.1,28.3,156.7,89.2,...,-1,1.245,4
...
```

### top_anomalies.csv
```
lep1_pt,lep2_pt,mll,met_et,...,prediction,anomaly_score
92.3,58.1,287.4,145.2,...,-1,2.103
103.5,64.2,312.1,167.8,...,-1,1.987
...
```

### summary_statistics.json
```json
{
  "total_events": 3000,
  "anomalies_detected": 154,
  "anomaly_rate": 5.13,
  "accuracy": 0.856,
  "precision": 0.812,
  "recall": 0.798,
  "f1_score": 0.805,
  "roc_auc": 0.923
}
```

---

## 🚀 Next Steps After Running

1. **Examine anomalies**: Check `top_anomalies.csv` for physics patterns
2. **Compare models**: Which models agree most?
3. **Adjust threshold**: Change contamination rate
4. **Visualize patterns**: Look at feature distributions
5. **Export for analysis**: Use CSV for further investigation
6. **Deploy in trigger**: Use StreamingAnomalyDetector for live data

---

## 📞 Tips & Tricks

### Faster Iteration
```python
# In Configuration cell, use smaller dataset
CONFIG['max_events'] = 1000
```

### Better Anomalies
```python
# Inject more anomalies for testing
CONFIG['contamination'] = 0.10  # 10% instead of 5%
```

### Custom Features
```python
# In Data Loading cell, add new physics variables
data['new_feature'] = compute_from_existing()
```

### Different Models
```python
# In Training cell, swap ensemble for single model
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
model.fit(X_train)
predictions = model.predict(X_test)
```

---

## ✅ Success Checklist

After running:
- [ ] Notebook runs without errors
- [ ] Data loads successfully
- [ ] Models train (you see epoch output)
- [ ] Predictions made (anomaly count shown)
- [ ] Visualizations appear
- [ ] Results exported to `outputs/` folder
- [ ] CSV files are readable

If all checked: **You're ready to analyze ATLAS data!**

---

*For detailed information, see COMPLETION_REPORT.md or SETUP_COMPLETE.md*
