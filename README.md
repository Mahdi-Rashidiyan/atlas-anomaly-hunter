# Real-Time Anomaly Detection for ATLAS Particle Detector Data

**A comprehensive machine learning system for detecting rare events in high-energy physics data**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project implements a sophisticated anomaly detection system for ATLAS Open Data from CERN's Large Hadron Collider. It uses multiple machine learning approaches including deep autoencoders, isolation forests, and one-class SVMs to identify rare physics events that could indicate new phenomena beyond the Standard Model.

**Key Features:**
- ✨ Ensemble anomaly detection with 4 complementary algorithms
- 🚀 Real-time streaming processing capability
- 📊 Comprehensive visualization and evaluation tools
- ⚡ Optimized for high-throughput particle physics data
- 🔬 Physics-aware feature engineering

---

## 🔬 Physics Context

### ATLAS Detector Data
This system analyzes collision events from the ATLAS detector at 13 TeV center-of-mass energy, specifically events with **exactly two leptons** (electrons or muons). These "dilepton" events are signatures of various physics processes including:

- **Standard Model Processes:**
  - Z boson decay (Z → ee, Z → μμ)
  - W boson pair production
  - Top quark pair production
  - Drell-Yan process

- **Potential Beyond Standard Model Signals:**
  - Z' bosons (heavy neutral gauge bosons)
  - Supersymmetric particles
  - Extra dimensions
  - Dark matter candidates

### Why Anomaly Detection?

Traditional physics analyses search for specific predicted signals. Anomaly detection offers a complementary **model-independent** approach that can:
- Discover unexpected new physics
- Find rare processes without prior modeling
- Reduce human bias in signal selection
- Enhance sensitivity to subtle deviations from Standard Model

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ ROOT Files │→ │ Uproot Parser│→ │ Feature Engineering│   │
│  └────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Anomaly Detection Layer                     │
│  ┌──────────────────┐  ┌─────────────────────────────┐     │
│  │ Ensemble Detector │  │  Individual Models:         │     │
│  │                   │  │  • Deep Autoencoder         │     │
│  │  • Model Fusion   │  │  • Isolation Forest         │     │
│  │  • Voting System  │  │  • One-Class SVM            │     │
│  │  • Confidence     │  │  • Elliptic Envelope        │     │
│  └──────────────────┘  └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Real-Time Processing & Monitoring               │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Event Stream │→ │ Online Learner│→│ Alert System     │   │
│  │ Simulator    │  │ (Welford's)  │  │ & Visualization  │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/atlas-anomaly-detection.git
cd atlas-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from atlas_anomaly_detector import ATLASDataLoader, EnsembleAnomalyDetector

# 1. Load ATLAS data
loader = ATLASDataLoader("path/to/data")
data = loader.prepare_data([
    "data_A.exactly2lep.root",
    "data_B.exactly2lep.root"
], max_events_per_file=50000)

# 2. Train ensemble detector
detector = EnsembleAnomalyDetector(contamination=0.05)
detector.fit(data.values)

# 3. Detect anomalies
predictions = detector.predict(test_data.values)
anomalies = test_data[predictions == -1]

print(f"Detected {len(anomalies)} anomalous events")
```

### Running the Demo

```bash
# Run complete analysis pipeline
python atlas_anomaly_detector.py

# Run real-time detection demo
python real_time_detector.py

# Run evaluation metrics
python evaluation_metrics.py
```

---

## 📊 Features & Algorithms

### 1. Deep Autoencoder
**Architecture:**
- Encoder: 14 → 64 → 32 → 16 → 8 (bottleneck)
- Decoder: 8 → 16 → 32 → 64 → 14
- Batch normalization and dropout for regularization
- MSE loss with reconstruction error as anomaly score

**Advantages:**
- Captures complex non-linear patterns
- Learns compressed representation of normal events
- Excellent for high-dimensional data

### 2. Isolation Forest
**Hyperparameters:**
- 200 trees
- Automatic sample sizing
- Parallel processing enabled

**Advantages:**
- Fast training and prediction
- Robust to outliers in training
- No assumptions about data distribution

### 3. One-Class SVM
**Configuration:**
- RBF kernel with auto gamma
- Nu parameter set to contamination rate

**Advantages:**
- Powerful for non-linear boundaries
- Strong theoretical foundation
- Good generalization

### 4. Elliptic Envelope
**Method:**
- Robust covariance estimation
- Mahalanobis distance scoring

**Advantages:**
- Interpretable (distance from center)
- Fast computation
- Works well for Gaussian-like distributions

### Ensemble Voting
- **Soft voting:** Majority decision (≥2 models)
- **Hard voting:** Unanimous agreement
- Individual model predictions available for analysis

---

## 🔬 Physics Features

### Engineered Variables

| Feature | Description | Physics Significance |
|---------|-------------|----------------------|
| `lep1_pt`, `lep2_pt` | Leading/subleading lepton transverse momentum | High pT → hard scattering |
| `lep_deltaR` | Angular separation Δ√(Δη² + Δφ²) | Topology of decay |
| `mll` | Dilepton invariant mass | Resonance peaks (Z, Z', etc.) |
| `met_et` | Missing transverse energy | Invisible particles (neutrinos, dark matter) |
| `jet_n`, `jet_ht` | Jet multiplicity and total pT | Hadronic activity |
| `centrality` | (ΣpT_lep + MET) / ΣpT_total | Event shape |

### Feature Distributions
Normal vs anomalous events show distinct patterns:
- **High invariant mass:** Could indicate Z' boson
- **High MET:** Possible SUSY or dark matter
- **Unusual jet configurations:** Multi-jet BSM processes
- **Extreme pT values:** Hard scattering processes

---

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy:** Overall correctness
- **Precision (Purity):** Fraction of detections that are true signals
- **Recall (Signal Efficiency):** Fraction of true signals detected
- **F1 Score:** Harmonic mean of precision and recall
- **ROC AUC:** Area under receiver operating characteristic curve
- **PR AUC:** Area under precision-recall curve

### Physics-Specific Metrics
- **Signal Efficiency:** Critical for discovery sensitivity
- **Background Rejection:** Important for reducing false positives
- **MCC:** Matthews Correlation Coefficient for imbalanced data

### Computational Performance
- **Throughput:** Events processed per second
- **Latency:** Processing time per event
- **Real-time capability:** Can process LHC data rates (kHz range)

---

## 🎨 Visualizations

The system generates comprehensive visualizations:

1. **Feature Distributions:** Compare normal vs anomalous events
2. **PCA Projection:** 2D visualization of high-dimensional data
3. **Model Agreement:** Heatmap showing inter-model consensus
4. **ROC Curves:** True positive vs false positive rates
5. **Precision-Recall Curves:** Precision vs recall trade-offs
6. **Confusion Matrices:** Detailed classification results
7. **Anomaly Scores:** Distribution of reconstruction errors

---

## 🔧 Configuration

### Hyperparameters

```python
# Ensemble Configuration
ensemble = EnsembleAnomalyDetector(
    contamination=0.05  # Expected fraction of anomalies (5%)
)

# Autoencoder Configuration
autoencoder = AutoencoderAnomalyDetector(
    encoding_dim=8,           # Bottleneck dimension
    contamination=0.05,
    epochs=50,                # Training epochs
    batch_size=256           # Batch size
)

# Real-Time Detector Configuration
streaming = StreamingAnomalyDetector(
    window_size=1000,         # Sliding window size
    update_frequency=100,     # Update model every N events
    contamination=0.05
)
```

---

## 📦 Project Structure

```
atlas-anomaly-detection/
├── atlas_anomaly_detector.py    # Main pipeline & ensemble
├── real_time_detector.py         # Streaming detection
├── evaluation_metrics.py         # Performance evaluation
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/                         # Data directory (not in repo)
│   ├── data_A.exactly2lep.root
│   └── mc_*.exactly2lep.root
├── models/                       # Saved models
│   └── ensemble_detector.pkl
├── outputs/                      # Results and plots
│   ├── feature_distributions.png
│   ├── pca_visualization.png
│   └── evaluation_report.txt
└── notebooks/                    # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

---

## 🎓 For CERN openlab Internship Application

### Project Highlights

This project demonstrates:

1. **Advanced Computing Skills:**
   - High-performance Python with NumPy/TensorFlow
   - Efficient data processing with Uproot/Awkward Array
   - Real-time streaming algorithms
   - Parallel processing and optimization

2. **Machine Learning Expertise:**
   - Deep learning (autoencoders)
   - Ensemble methods
   - Online learning algorithms
   - Model evaluation and validation

3. **Physics Understanding:**
   - ATLAS detector and data formats
   - Particle physics feature engineering
   - BSM signal characteristics
   - Statistical methods in HEP

4. **Software Engineering:**
   - Clean, modular code architecture
   - Comprehensive documentation
   - Version control (Git)
   - Reproducible research practices

### Future Enhancements

**Potential improvements for openlab project:**

1. **GPU Acceleration:** Port algorithms to CUDA for faster processing
2. **Distributed Computing:** Scale to full ATLAS dataset using Spark/Dask
3. **Active Learning:** Incorporate physicist feedback to improve detector
4. **Interpretability:** Add SHAP/LIME for explaining anomaly predictions
5. **Integration:** Connect to ATLAS trigger system for live deployment
6. **New Architectures:** Experiment with graph neural networks, transformers
7. **Multi-Modal:** Incorporate calorimeter images and tracking information

---

## 📚 References

### ATLAS Open Data
- [ATLAS Open Data Portal](https://opendata.atlas.cern/)
- [Dataset Documentation](http://opendata.atlas.cern/release/2020/documentation/)
- [ATLAS Collaboration Papers](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/)

### Anomaly Detection in HEP
- "Learning New Physics from a Machine" (Baldi et al., 2018)
- "Anomaly Detection in High Energy Physics" (Nachman & Shih, 2020)
- "Autoencoders for Unsupervised Anomaly Detection" (Roy et al., 2019)

### CERN openlab
- [CERN openlab Website](https://openlab.cern/)
- [Application Portal](https://careers.cern/students)
- [Past Projects](https://openlab.cern/education)

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional anomaly detection algorithms
- Performance optimizations
- New visualization methods
- Documentation improvements
- Bug fixes

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

**Note:** ATLAS Open Data is released under CC0 1.0 Universal license.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@university.edu
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

*Prepared for CERN openlab Summer Student Programme Application*

---

## 🙏 Acknowledgments

- ATLAS Collaboration for providing open data
- CERN openlab for inspiration
- Scikit-learn, TensorFlow, and Uproot communities
- All contributors to open-source HEP software

---

## 📞 Contact & Support

For questions about this project or CERN openlab applications:
- Create an issue on GitHub
- Email: your.email@university.edu
- CERN openlab: openlab-admins@cern.ch

**Good luck with your application! 🚀**