"""
Real-Time Streaming Anomaly Detector for ATLAS Data
Processes events in real-time with minimal latency
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import time
import threading
from queue import Queue, Empty


class StreamingAnomalyDetector:
    """
    Real-time streaming anomaly detector using pre-trained ensemble models
    Designed for low-latency online anomaly detection
    """
    
    def __init__(self, ensemble_detector, buffer_size: int = 1000, batch_size: int = 100):
        """
        Initialize streaming detector
        
        Args:
            ensemble_detector: Fitted EnsembleAnomalyDetector instance
            buffer_size: Maximum size of event buffer
            batch_size: Batch size for processing
        """
        self.ensemble = ensemble_detector
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Event buffer
        self.event_buffer = deque(maxlen=buffer_size)
        self.predictions_buffer = deque(maxlen=buffer_size)
        self.scores_buffer = deque(maxlen=buffer_size)
        
        # Processing queue
        self.event_queue = Queue()
        self.result_queue = Queue()
        
        # Statistics
        self.total_events = 0
        self.total_anomalies = 0
        self.processing_times = deque(maxlen=100)
        
        # Thread control
        self.running = False
        self.worker_thread = None
    
    def process_event(self, event: np.ndarray) -> Dict:
        """
        Process a single event with minimal latency
        
        Args:
            event: Feature vector for single event
            
        Returns:
            Dictionary with prediction and score
        """
        start_time = time.time()
        
        # Ensure correct shape
        if event.ndim == 1:
            event = event.reshape(1, -1)
        
        # Get prediction and scores
        prediction = self.ensemble.predict(event)[0]
        individual_preds = self.ensemble.get_individual_predictions(event)
        autoencoder_score = self.ensemble.models['autoencoder'].anomaly_scores(event)[0]
        
        # Model agreement
        agreement = sum([1 for p in individual_preds.values() if p[0] == -1])
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        self.processing_times.append(processing_time)
        
        # Update statistics
        self.total_events += 1
        if prediction == -1:
            self.total_anomalies += 1
        
        # Add to buffers
        self.event_buffer.append(event[0])
        self.predictions_buffer.append(prediction)
        self.scores_buffer.append(autoencoder_score)
        
        result = {
            'event_id': self.total_events,
            'prediction': prediction,  # 1 = normal, -1 = anomaly
            'is_anomaly': prediction == -1,
            'anomaly_score': float(autoencoder_score),
            'model_agreement': agreement,  # 0-4 models voting anomaly
            'processing_time_ms': processing_time,
            'individual_predictions': {k: int(v[0]) for k, v in individual_preds.items()}
        }
        
        return result
    
    def process_batch(self, events: np.ndarray) -> List[Dict]:
        """
        Process a batch of events
        
        Args:
            events: Array of feature vectors (batch_size, n_features)
            
        Returns:
            List of result dictionaries
        """
        start_time = time.time()
        
        # Get predictions
        predictions = self.ensemble.predict(events)
        individual_preds = self.ensemble.get_individual_predictions(events)
        autoencoder_scores = self.ensemble.models['autoencoder'].anomaly_scores(events)
        
        # Model agreement
        agreements = np.sum([p == -1 for p in individual_preds.values()], axis=0)
        
        # Processing time
        total_time = (time.time() - start_time) * 1000  # milliseconds
        per_event_time = total_time / len(events)
        
        # Update statistics
        self.total_events += len(events)
        self.total_anomalies += np.sum(predictions == -1)
        
        # Add to buffers
        for event in events:
            self.event_buffer.append(event)
        for pred in predictions:
            self.predictions_buffer.append(pred)
        for score in autoencoder_scores:
            self.scores_buffer.append(score)
        
        # Create results
        results = []
        for i in range(len(events)):
            individual = {k: int(v[i]) for k, v in individual_preds.items()}
            result = {
                'event_id': self.total_events - len(events) + i + 1,
                'prediction': int(predictions[i]),
                'is_anomaly': bool(predictions[i] == -1),
                'anomaly_score': float(autoencoder_scores[i]),
                'model_agreement': int(agreements[i]),
                'processing_time_ms': per_event_time,
                'individual_predictions': individual
            }
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get current streaming statistics
        
        Returns:
            Dictionary of statistics
        """
        anomaly_rate = (self.total_anomalies / self.total_events * 100) if self.total_events > 0 else 0
        
        avg_processing_time = (np.mean(self.processing_times) 
                               if len(self.processing_times) > 0 else 0)
        max_processing_time = (np.max(self.processing_times) 
                               if len(self.processing_times) > 0 else 0)
        min_processing_time = (np.min(self.processing_times) 
                               if len(self.processing_times) > 0 else 0)
        
        throughput = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            'total_events_processed': self.total_events,
            'total_anomalies_detected': self.total_anomalies,
            'anomaly_rate_percent': anomaly_rate,
            'avg_processing_time_ms': avg_processing_time,
            'min_processing_time_ms': min_processing_time,
            'max_processing_time_ms': max_processing_time,
            'throughput_events_per_sec': throughput,
            'buffer_utilization': len(self.event_buffer) / self.buffer_size
        }
    
    def get_recent_anomalies(self, n: int = 10) -> List[Dict]:
        """
        Get the N most recent anomalous events
        
        Args:
            n: Number of recent anomalies to return
            
        Returns:
            List of anomalous events with details
        """
        recent_events = list(self.event_buffer)
        recent_preds = list(self.predictions_buffer)
        recent_scores = list(self.scores_buffer)
        
        # Find anomalies
        anomalies = []
        for i, pred in enumerate(recent_preds):
            if pred == -1:
                anomalies.append({
                    'index': i,
                    'event': recent_events[i],
                    'score': recent_scores[i],
                    'event_id': self.total_events - len(recent_preds) + i + 1
                })
        
        # Return most recent
        return sorted(anomalies, key=lambda x: x['index'], reverse=True)[:n]
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.total_events = 0
        self.total_anomalies = 0
        self.processing_times.clear()
        self.event_buffer.clear()
        self.predictions_buffer.clear()
        self.scores_buffer.clear()
    
    def start_worker(self):
        """Start background worker thread for async processing"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
    
    def stop_worker(self):
        """Stop background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker_loop(self):
        """Background worker loop for processing events from queue"""
        while self.running:
            try:
                # Get batch of events from queue
                batch = []
                try:
                    # Try to fill a batch
                    for _ in range(self.batch_size):
                        event = self.event_queue.get(timeout=0.1)
                        batch.append(event)
                except Empty:
                    pass
                
                if batch:
                    # Process batch
                    events_array = np.array(batch)
                    results = self.process_batch(events_array)
                    
                    # Put results in result queue
                    for result in results:
                        self.result_queue.put(result)
                
            except Exception as e:
                print(f"Error in worker loop: {e}")
    
    def queue_event(self, event: np.ndarray):
        """
        Queue an event for async processing
        
        Args:
            event: Feature vector for single event
        """
        self.event_queue.put(event)
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get result from async processing queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None


class AdaptiveAnomalyDetector(StreamingAnomalyDetector):
    """
    Adaptive streaming detector that updates thresholds based on recent data
    Useful for handling distribution drift
    """
    
    def __init__(self, ensemble_detector, buffer_size: int = 1000, 
                 adaptation_window: int = 100):
        """
        Initialize adaptive detector
        
        Args:
            ensemble_detector: Fitted EnsembleAnomalyDetector instance
            buffer_size: Maximum size of event buffer
            adaptation_window: Number of events for threshold adaptation
        """
        super().__init__(ensemble_detector, buffer_size)
        self.adaptation_window = adaptation_window
        self.score_history = deque(maxlen=adaptation_window)
        self.adaptive_threshold = None
        self.base_contamination = ensemble_detector.contamination
    
    def process_event(self, event: np.ndarray) -> Dict:
        """Process event with adaptive thresholding"""
        result = super().process_event(event)
        
        # Track score for adaptation
        self.score_history.append(result['anomaly_score'])
        
        # Periodically update threshold
        if self.total_events % self.adaptation_window == 0 and len(self.score_history) > 0:
            self._update_adaptive_threshold()
        
        return result
    
    def _update_adaptive_threshold(self):
        """Update anomaly threshold based on recent score distribution"""
        if len(self.score_history) > 0:
            scores = np.array(list(self.score_history))
            # Set threshold to expected contamination percentile
            percentile = 100 * (1 - self.base_contamination)
            self.adaptive_threshold = np.percentile(scores, percentile)
    
    def get_statistics(self) -> Dict:
        """Get statistics including adaptive threshold info"""
        stats = super().get_statistics()
        stats['adaptive_threshold'] = self.adaptive_threshold
        stats['score_history_length'] = len(self.score_history)
        return stats
