import time
import asyncio
from typing import Dict

class ApiBudgeter:
    """
    TokenBucketBudgeter: A strict rate limiter for API calls.
    
    Binance Limits:
    - 2400 weight per minute (IP limit)
    - 1200 orders per minute (Order limit)
    
    Strategy:
    - Target: 1800 weight/min (75% of limit) to be safe.
    - Rate: 30 weight/second.
    - Bucket Capacity: 100 weight (allow small bursts).
    
    Usage:
    - Call `await wait_for_token(cost)` before EVERY API call.
    - This creates a smooth stream of requests, preventing "queue full" errors.
    """
    def __init__(self, target_weight_per_min: int = 1800, max_bucket_size: int = 10):
        self.rate = target_weight_per_min / 60.0  # Tokens per second (e.g., 30.0)
        self.capacity = float(max_bucket_size)    # Max burst size (REDUCED to 10 to smooth out CCXT queue)
        self.tokens = float(max_bucket_size)      # Current tokens
        self.last_update = time.monotonic()
        
        # Stats for monitoring
        self.total_consumed = 0
        self.wait_time_total = 0.0
        self.buckets: Dict[str, int] = {} # Usage by category

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)

    def can_spend(self, cost: int) -> bool:
        """Check if we can spend without waiting (synchronous check)."""
        self._refill()
        return self.tokens >= cost

    async def wait_for_token(self, cost: int = 1, category: str = "other"):
        """
        Blocking async call that waits until enough tokens are available.
        """
        while True:
            self._refill()
            
            if self.tokens >= cost:
                self.tokens -= cost
                self.total_consumed += cost
                self.buckets[category] = self.buckets.get(category, 0) + cost
                return
            
            # Calculate needed tokens and wait time
            needed = cost - self.tokens
            wait_seconds = needed / self.rate
            
            # Add a tiny buffer to ensure we have enough on wake up
            wait_seconds = max(0.01, wait_seconds + 0.01)
            
            self.wait_time_total += wait_seconds
            await asyncio.sleep(wait_seconds)

    def remaining_capacity(self) -> float:
        """Return current available tokens (for logging/decisions)."""
        self._refill()
        return self.tokens

    def summary(self) -> str:
        """Return usage stats string."""
        self._refill()
        # Sort buckets by usage
        usage_str = " ".join([f"{k}:{v}" for k, v in sorted(self.buckets.items(), key=lambda x: x[1], reverse=True)])
        return f"Budget: {self.tokens:.1f}/{self.capacity} (Rate: {self.rate:.1f}/s) | Total: {self.total_consumed} | Waited: {self.wait_time_total:.1f}s | {usage_str}"
