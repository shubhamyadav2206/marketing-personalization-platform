"""
Monitoring and observability module for the pipeline.
"""
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import threading

logger = logging.getLogger(__name__)

# Prometheus metrics
pipeline_runs_total = Counter('pipeline_runs_total', 'Total number of pipeline runs', ['status'])
pipeline_duration_seconds = Histogram('pipeline_duration_seconds', 'Pipeline execution duration in seconds', ['step'])
pipeline_errors_total = Counter('pipeline_errors_total', 'Total number of pipeline errors', ['error_type'])
pipeline_records_processed = Counter('pipeline_records_processed_total', 'Total records processed', ['step'])
pipeline_anomalies_detected = Gauge('pipeline_anomalies_detected', 'Number of anomalies detected', ['anomaly_type'])

api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
api_request_duration_seconds = Histogram('api_request_duration_seconds', 'API request duration in seconds', ['endpoint'])
api_recommendations_generated = Counter('api_recommendations_generated_total', 'Total recommendations generated', ['user_id'])


class PipelineMonitor:
    """Monitor pipeline execution and track metrics."""
    
    def __init__(self, output_path: str = None):
        self.output_path = output_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "reports",
            "pipeline_metrics.json"
        )
        self.metrics = {
            "pipeline_runs": [],
            "step_metrics": defaultdict(list),
            "anomalies": [],
            "errors": []
        }
        self.current_run = None
        self.lock = threading.Lock()
    
    def start_run(self, run_id: str, input_path: str = None):
        """Start tracking a pipeline run."""
        with self.lock:
            self.current_run = {
                "run_id": run_id,
                "start_time": datetime.utcnow().isoformat(),
                "input_path": input_path,
                "steps": [],
                "status": "running"
            }
            pipeline_runs_total.labels(status='running').inc()
    
    def record_step(self, step_name: str, duration: float, input_count: int, output_count: int, 
                   errors: List[str] = None, anomalies: Dict = None):
        """Record a pipeline step."""
        with self.lock:
            if self.current_run is None:
                self.start_run(f"auto_{datetime.utcnow().isoformat()}")
            
            step_metric = {
                "step_name": step_name,
                "duration_seconds": duration,
                "input_count": input_count,
                "output_count": output_count,
                "throughput_records_per_second": output_count / duration if duration > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if errors:
                step_metric["errors"] = errors
                pipeline_errors_total.labels(error_type=step_name).inc(len(errors))
            
            if anomalies:
                step_metric["anomalies"] = anomalies
                for anomaly_type, count in anomalies.items():
                    if count > 0:
                        pipeline_anomalies_detected.labels(anomaly_type=anomaly_type).set(count)
            
            self.current_run["steps"].append(step_metric)
            self.metrics["step_metrics"][step_name].append(step_metric)
            pipeline_duration_seconds.labels(step=step_name).observe(duration)
            pipeline_records_processed.labels(step=step_name).inc(output_count)
    
    def end_run(self, status: str = "completed", error: str = None):
        """End tracking a pipeline run."""
        with self.lock:
            if self.current_run is None:
                return
            
            self.current_run["end_time"] = datetime.utcnow().isoformat()
            self.current_run["status"] = status
            
            start = datetime.fromisoformat(self.current_run["start_time"].replace('Z', '+00:00'))
            end = datetime.utcnow()
            self.current_run["total_duration_seconds"] = (end - start.replace(tzinfo=None)).total_seconds()
            
            if error:
                self.current_run["error"] = error
                self.metrics["errors"].append({
                    "run_id": self.current_run["run_id"],
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            self.metrics["pipeline_runs"].append(self.current_run)
            
            # Keep only last 100 runs
            if len(self.metrics["pipeline_runs"]) > 100:
                self.metrics["pipeline_runs"] = self.metrics["pipeline_runs"][-100:]
            
            pipeline_runs_total.labels(status=status).inc()
            self.current_run = None
    
    def detect_latency_anomalies(self, threshold_multiplier: float = 2.0) -> List[Dict]:
        """Detect latency anomalies based on historical data."""
        anomalies = []
        
        with self.lock:
            for step_name, step_history in self.metrics["step_metrics"].items():
                if len(step_history) < 2:
                    continue
                
                # Calculate average duration
                durations = [s["duration_seconds"] for s in step_history[-10:]]  # Last 10 runs
                avg_duration = sum(durations) / len(durations)
                std_duration = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5
                threshold = avg_duration + (threshold_multiplier * std_duration)
                
                # Check latest run
                latest = step_history[-1]
                if latest["duration_seconds"] > threshold:
                    anomalies.append({
                        "step_name": step_name,
                        "duration": latest["duration_seconds"],
                        "expected_max": threshold,
                        "deviation": latest["duration_seconds"] - threshold,
                        "timestamp": latest["timestamp"]
                    })
        
        return anomalies
    
    def get_summary(self) -> Dict:
        """Get summary of pipeline metrics."""
        with self.lock:
            if not self.metrics["pipeline_runs"]:
                return {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "average_duration": 0.0,
                    "total_records_processed": 0
                }
            
            runs = self.metrics["pipeline_runs"]
            successful = sum(1 for r in runs if r["status"] == "completed")
            failed = sum(1 for r in runs if r["status"] == "failed")
            avg_duration = sum(r.get("total_duration_seconds", 0) for r in runs) / len(runs)
            
            # Calculate total records processed
            total_records = 0
            for run in runs:
                for step in run.get("steps", []):
                    total_records += step.get("output_count", 0)
            
            return {
                "total_runs": len(runs),
                "successful_runs": successful,
                "failed_runs": failed,
                "success_rate": successful / len(runs) if runs else 0.0,
                "average_duration_seconds": avg_duration,
                "total_records_processed": total_records,
                "last_run": runs[-1] if runs else None
            }
    
    def save_metrics(self):
        """Save metrics to file."""
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            summary = self.get_summary()
            metrics_data = {
                "summary": summary,
                "pipeline_runs": self.metrics["pipeline_runs"][-20:],  # Last 20 runs
                "timestamp": datetime.utcnow().isoformat()
            }
            
            with open(self.output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


class APIMonitor:
    """Monitor API requests and performance."""
    
    def __init__(self):
        self.request_logs = []
        self.lock = threading.Lock()
    
    def log_request(self, endpoint: str, method: str, status_code: int, duration: float, 
                   user_id: str = None, **kwargs):
        """Log an API request."""
        with self.lock:
            log_entry = {
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_seconds": duration,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs
            }
            
            self.request_logs.append(log_entry)
            
            # Keep only last 1000 requests
            if len(self.request_logs) > 1000:
                self.request_logs = self.request_logs[-1000:]
            
            # Update Prometheus metrics
            api_requests_total.labels(endpoint=endpoint, status=str(status_code)).inc()
            api_request_duration_seconds.labels(endpoint=endpoint).observe(duration)
            
            if user_id:
                api_recommendations_generated.labels(user_id=user_id).inc()
    
    def get_latency_stats(self, endpoint: str = None, window_minutes: int = 60) -> Dict:
        """Get latency statistics for requests."""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            filtered_logs = [
                log for log in self.request_logs
                if datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) >= cutoff_time.replace(tzinfo=None)
            ]
            
            if endpoint:
                filtered_logs = [log for log in filtered_logs if log["endpoint"] == endpoint]
            
            if not filtered_logs:
                return {
                    "count": 0,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0
                }
            
            latencies = [log["duration_seconds"] * 1000 for log in filtered_logs]  # Convert to ms
            latencies.sort()
            
            return {
                "count": len(latencies),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": latencies[len(latencies) // 2] if latencies else 0.0,
                "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
                "p99_latency_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0.0
            }


# Global monitor instances
_pipeline_monitor = None
_api_monitor = None

def get_pipeline_monitor() -> PipelineMonitor:
    """Get or create pipeline monitor instance."""
    global _pipeline_monitor
    if _pipeline_monitor is None:
        _pipeline_monitor = PipelineMonitor()
    return _pipeline_monitor

def get_api_monitor() -> APIMonitor:
    """Get or create API monitor instance."""
    global _api_monitor
    if _api_monitor is None:
        _api_monitor = APIMonitor()
    return _api_monitor
