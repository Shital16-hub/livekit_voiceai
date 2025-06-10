"""
Performance monitoring for LiveKit RAG Agent
Tracks latency and ensures sub-2-second response times
"""
import time
import asyncio
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
from contextlib import asynccontextmanager
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track agent performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.session_metrics: Dict[str, list] = {
            "stt_latency": [],
            "rag_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "total_latency": []
        }
        self.threshold_ms = config.target_latency_ms
        
    @asynccontextmanager
    async def time_operation(self, operation_name: str):
        """Context manager to time operations"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            self.metrics[operation_name] = duration_ms
            
            # Add to session metrics if it's a pipeline component
            if operation_name.endswith("_latency") and operation_name in self.session_metrics:
                self.session_metrics[operation_name].append(duration_ms)
            
            if config.log_performance:
                status = "✅" if duration_ms < self._get_threshold(operation_name) else "⚠️"
                logger.info(f"{status} {operation_name}: {duration_ms:.1f}ms")
    
    def _get_threshold(self, operation_name: str) -> float:
        """Get performance threshold for specific operations"""
        thresholds = {
            "stt_latency": 400,
            "rag_latency": 400,
            "llm_latency": 1000,
            "tts_latency": 600,
            "total_latency": self.threshold_ms
        }
        return thresholds.get(operation_name, 1000)
    
    async def time_async_function(self, func: Callable, operation_name: str, *args, **kwargs) -> Any:
        """Time an async function execution"""
        async with self.time_operation(operation_name):
            result = await func(*args, **kwargs)
        return result
    
    def get_total_latency(self) -> float:
        """Calculate total pipeline latency"""
        pipeline_components = ["stt_latency", "rag_latency", "llm_latency", "tts_latency"]
        total = sum(self.metrics.get(component, 0) for component in pipeline_components)
        self.metrics["total_latency"] = total
        return total
    
    def check_performance_targets(self) -> Dict[str, bool]:
        """Check if performance targets are met"""
        results = {}
        total_latency = self.get_total_latency()
        
        results["total_under_target"] = total_latency < self.threshold_ms
        results["rag_under_target"] = self.metrics.get("rag_latency", 0) < config.rag_timeout_ms
        
        if not results["total_under_target"]:
            logger.warning(f"⚠️ Total latency {total_latency:.1f}ms exceeds target {self.threshold_ms}ms")
        
        if not results["rag_under_target"]:
            logger.warning(f"⚠️ RAG latency {self.metrics.get('rag_latency', 0):.1f}ms exceeds target {config.rag_timeout_ms}ms")
        
        return results
    
    def get_session_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of session performance"""
        stats = {}
        
        for metric_name, values in self.session_metrics.items():
            if values:
                stats[metric_name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                stats[metric_name] = {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        return stats
    
    def log_session_summary(self):
        """Log session performance summary"""
        stats = self.get_session_stats()
        logger.info("📊 Session Performance Summary:")
        
        for metric_name, metric_stats in stats.items():
            if metric_stats["count"] > 0:
                logger.info(f"  {metric_name}: avg={metric_stats['avg']:.1f}ms, "
                           f"min={metric_stats['min']:.1f}ms, "
                           f"max={metric_stats['max']:.1f}ms, "
                           f"count={metric_stats['count']}")
    
    def reset_metrics(self):
        """Reset current metrics"""
        self.metrics.clear()
    
    def reset_session(self):
        """Reset session metrics"""
        for metric_list in self.session_metrics.values():
            metric_list.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Decorator for timing functions
def time_function(operation_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await performance_monitor.time_async_function(func, operation_name, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    performance_monitor.metrics[operation_name] = duration_ms
                    if config.log_performance:
                        logger.info(f"⏱️ {operation_name}: {duration_ms:.1f}ms")
            return sync_wrapper
    return decorator

# Context manager for easy timing
@asynccontextmanager
async def time_operation(operation_name: str):
    """Easy-to-use context manager for timing operations"""
    async with performance_monitor.time_operation(operation_name):
        yield

# Performance checker
async def check_latency_target():
    """Check if current latency meets targets"""
    total_latency = performance_monitor.get_total_latency()
    
    if total_latency > config.target_latency_ms:
        logger.error(f"❌ Latency target missed: {total_latency:.1f}ms > {config.target_latency_ms}ms")
        return False
    else:
        logger.info(f"✅ Latency target met: {total_latency:.1f}ms < {config.target_latency_ms}ms")
        return True

if __name__ == "__main__":
    # Test the performance monitor
    import asyncio
    
    async def test_monitor():
        async with time_operation("test_operation"):
            await asyncio.sleep(0.1)  # Simulate 100ms operation
        
        logger.info(f"Test operation took: {performance_monitor.metrics['test_operation']:.1f}ms")
    
    asyncio.run(test_monitor())