"""
Evaluation Metrics and Performance Analysis for Anomaly Detection
Includes physics-specific metrics and benchmarking tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import time


class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluation of anomaly detection performance
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None):
        """
        Args:
            y_true: True labels (1 for normal, -1 for anomaly)
            y_pred: Predicted labels (1 for normal, -1 for anomaly)
            y_scores: Anomaly scores (optional, for ROC/PR curves)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        
        # Convert labels to binary (0 = normal, 1 = anomaly)
        self.y_true_binary = (y_true == -1).astype(int)
        self.y_pred_binary = (y_pred == -1).astype(int)
    
    def compute_metrics(self) -> Dict:
        """
        Compute comprehensive metrics
        
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true_binary, self.y_pred_binary).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also called sensitivity or TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
        
        # Physics-specific metrics
        metrics = {
            # Basic classification metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'mcc': mcc,
            
            # Error rates
            'false_positive_rate': fpr,
            'false_negative_rate': 1 - recall,
            
            # Confusion matrix
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Counts
            'total_anomalies': int(np.sum(self.y_true_binary)),
            'detected_anomalies': int(np.sum(self.y_pred_binary)),
        }
        
        # ROC AUC if scores available
        if self.y_scores is not None:
            fpr_curve, tpr_curve, _ = roc_curve(self.y_true_binary, self.y_scores)
            metrics['roc_auc'] = auc(fpr_curve, tpr_curve)
            
            # PR AUC
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_true_binary, self.y_scores)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        # Physics interpretation
        metrics['signal_efficiency'] = recall  # Fraction of true signals detected
        metrics['background_rejection'] = specificity  # Fraction of background correctly identified
        metrics['purity'] = precision  # Fraction of detected events that are true signals
        
        return metrics
    
    def print_report(self):
        """Print detailed evaluation report"""
        metrics = self.compute_metrics()
        
        print("=" * 80)
        print("ANOMALY DETECTION EVALUATION REPORT")
        print("=" * 80)
        
        print("\n📊 CLASSIFICATION METRICS")
        print("-" * 80)
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1 Score:        {metrics['f1_score']:.4f}")
        print(f"Specificity:     {metrics['specificity']:.4f}")
        print(f"MCC:             {metrics['mcc']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:         {metrics['roc_auc']:.4f}")
            print(f"PR AUC:          {metrics['pr_auc']:.4f}")
        
        print("\n🎯 PHYSICS INTERPRETATION")
        print("-" * 80)
        print(f"Signal Efficiency:      {metrics['signal_efficiency']:.4f} "
              f"({metrics['signal_efficiency']*100:.2f}% of true signals detected)")
        print(f"Background Rejection:   {metrics['background_rejection']:.4f} "
              f"({metrics['background_rejection']*100:.2f}% of background rejected)")
        print(f"Purity:                 {metrics['purity']:.4f} "
              f"({metrics['purity']*100:.2f}% of detections are true signals)")
        
        print("\n📈 ERROR ANALYSIS")
        print("-" * 80)
        print(f"False Positive Rate:    {metrics['false_positive_rate']:.4f} "
              f"({metrics['false_positive_rate']*100:.2f}% of normal events misclassified)")
        print(f"False Negative Rate:    {metrics['false_negative_rate']:.4f} "
              f"({metrics['false_negative_rate']*100:.2f}% of anomalies missed)")
        
        print("\n🔢 CONFUSION MATRIX")
        print("-" * 80)
        print(f"True Positives:         {metrics['true_positives']:6d}")
        print(f"True Negatives:         {metrics['true_negatives']:6d}")
        print(f"False Positives:        {metrics['false_positives']:6d}")
        print(f"False Negatives:        {metrics['false_negatives']:6d}")
        
        print("\n📊 SUMMARY")
        print("-" * 80)
        print(f"Total Events:           {len(self.y_true):6d}")
        print(f"True Anomalies:         {metrics['total_anomalies']:6d} "
              f"({metrics['total_anomalies']/len(self.y_true)*100:.2f}%)")
        print(f"Detected Anomalies:     {metrics['detected_anomalies']:6d} "
              f"({metrics['detected_anomalies']/len(self.y_true)*100:.2f}%)")
        
        print("\n" + "=" * 80)
    
    def plot_roc_curve(self, title: str = "ROC Curve") -> plt.Figure:
        """Plot ROC curve"""
        if self.y_scores is None:
            raise ValueError("Scores required for ROC curve")
        
        fpr, tpr, thresholds = roc_curve(self.y_true_binary, self.y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, title: str = "Precision-Recall Curve") -> plt.Figure:
        """Plot Precision-Recall curve"""
        if self.y_scores is None:
            raise ValueError("Scores required for PR curve")
        
        precision, recall, thresholds = precision_recall_curve(self.y_true_binary, self.y_scores)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        
        # Baseline (random classifier)
        baseline = np.sum(self.y_true_binary) / len(self.y_true_binary)
        ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
                label=f'Random (Precision = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Signal Efficiency)', fontsize=12)
        ax.set_ylabel('Precision (Purity)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true_binary, self.y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_score_distribution(self, title: str = "Anomaly Score Distribution") -> plt.Figure:
        """Plot distribution of anomaly scores"""
        if self.y_scores is None:
            raise ValueError("Scores required for distribution plot")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separate scores by true label
        normal_scores = self.y_scores[self.y_true_binary == 0]
        anomaly_scores = self.y_scores[self.y_true_binary == 1]
        
        # Plot distributions
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class PerformanceBenchmark:
    """
    Benchmark computational performance
    """
    
    def __init__(self):
        self.results = []
    
    def benchmark_model(self, 
                       model, 
                       X_test: np.ndarray, 
                       model_name: str,
                       n_trials: int = 5) -> Dict:
        """
        Benchmark model performance
        
        Args:
            model: Model to benchmark
            X_test: Test data
            model_name: Name for reporting
            n_trials: Number of trials for timing
            
        Returns:
            Benchmark results
        """
        print(f"\n[BENCHMARK] Testing {model_name}...")
        
        # Warmup
        _ = model.predict(X_test[:100])
        
        # Time predictions
        times = []
        for _ in range(n_trials):
            start = time.time()
            predictions = model.predict(X_test)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(X_test) / avg_time
        latency_per_event = (avg_time / len(X_test)) * 1000  # ms
        
        result = {
            'model': model_name,
            'n_samples': len(X_test),
            'avg_time_sec': avg_time,
            'std_time_sec': std_time,
            'throughput_events_per_sec': throughput,
            'latency_ms_per_event': latency_per_event
        }
        
        self.results.append(result)
        
        print(f"  ✓ Average time: {avg_time:.4f} ± {std_time:.4f} sec")
        print(f"  ✓ Throughput: {throughput:.1f} events/sec")
        print(f"  ✓ Latency: {latency_per_event:.4f} ms/event")
        
        return result
    
    def plot_benchmark_results(self) -> plt.Figure:
        """Plot benchmark comparison"""
        if not self.results:
            raise ValueError("No benchmark results to plot")
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Throughput comparison
        axes[0].barh(df['model'], df['throughput_events_per_sec'], color='steelblue')
        axes[0].set_xlabel('Throughput (events/second)', fontsize=12)
        axes[0].set_title('Model Throughput Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Latency comparison
        axes[1].barh(df['model'], df['latency_ms_per_event'], color='coral')
        axes[1].set_xlabel('Latency (ms/event)', fontsize=12)
        axes[1].set_title('Model Latency Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """Generate benchmark report"""
        if not self.results:
            print("No benchmark results available")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        print(f"\nTest Dataset Size: {self.results[0]['n_samples']} events")
        print("\nResults:")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"\n{row['model']}:")
            print(f"  Throughput:  {row['throughput_events_per_sec']:10.1f} events/sec")
            print(f"  Latency:     {row['latency_ms_per_event']:10.4f} ms/event")
            print(f"  Total Time:  {row['avg_time_sec']:10.4f} ± {row['std_time_sec']:.4f} sec")
        
        # Find best performing
        best_throughput = df.loc[df['throughput_events_per_sec'].idxmax()]
        best_latency = df.loc[df['latency_ms_per_event'].idxmin()]
        
        print("\n" + "-" * 80)
        print(f"🏆 Best Throughput: {best_throughput['model']} "
              f"({best_throughput['throughput_events_per_sec']:.1f} events/sec)")
        print(f"🏆 Best Latency:    {best_latency['model']} "
              f"({best_latency['latency_ms_per_event']:.4f} ms/event)")
        print("=" * 80)


def demo_evaluation():
    """Demonstration of evaluation tools"""
    # Generate synthetic results
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (5% anomalies)
    y_true = np.ones(n_samples)
    anomaly_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    y_true[anomaly_indices] = -1
    
    # Predicted labels (with some errors)
    y_pred = y_true.copy()
    # Add false positives
    fp_indices = np.random.choice(np.where(y_true == 1)[0], 20, replace=False)
    y_pred[fp_indices] = -1
    # Add false negatives
    fn_indices = np.random.choice(np.where(y_true == -1)[0], 5, replace=False)
    y_pred[fn_indices] = 1
    
    # Anomaly scores
    y_scores = np.random.randn(n_samples)
    y_scores[y_true == -1] += 2  # Anomalies have higher scores
    
    # Evaluate
    evaluator = AnomalyDetectionEvaluator(y_true, y_pred, y_scores)
    evaluator.print_report()
    
    # Generate plots
    fig1 = evaluator.plot_roc_curve()
    plt.savefig('evaluation_roc.png', dpi=150, bbox_inches='tight')
    
    fig2 = evaluator.plot_precision_recall_curve()
    plt.savefig('evaluation_pr.png', dpi=150, bbox_inches='tight')
    
    fig3 = evaluator.plot_confusion_matrix()
    plt.savefig('evaluation_cm.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ Evaluation plots saved!")


if __name__ == "__main__":
    demo_evaluation()