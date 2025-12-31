import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass(frozen=True)
class ApiGovernorConfig:
    """
    Two-dimensional governor:
    - Token bucket (rate budget): limits API weight per time.
    - In-flight semaphore (concurrency budget): prevents CCXT queue overflow.
    - Exit reserve: keeps a small token reserve that only "exit" lane can use.
    """

    # Rate budget (token bucket)
    target_weight_per_min: int = 1800
    burst: int = 10
    exit_token_reserve: int = 3

    # Concurrency budget (in-flight)
    max_inflight: int = 25

    # Safety timeout per call (avoid hangs)
    call_timeout_sec: float = 3.0


class ApiGovernor:
    """
    Precision API gate: every call must pass through this method.

    Contract:
      call executes iff tokens >= cost and inflight < max_inflight.
      Non-exit lanes may not consume into the exit reserve.
    """

    def __init__(self, cfg: ApiGovernorConfig):
        self.cfg = cfg
        self._rate_per_sec = float(cfg.target_weight_per_min) / 60.0
        self._capacity = float(cfg.burst)
        self._tokens = float(cfg.burst)
        self._last_refill = time.monotonic()

        self._token_lock = asyncio.Lock()
        self._inflight = asyncio.Semaphore(cfg.max_inflight)

        # Minimal stats (debuggable but not spammy)
        self.total_consumed = 0
        self.total_wait_sec = 0.0
        
        # EMERGENCY BRAKE
        self._circuit_broken = False

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        if elapsed <= 0:
            return
        
        # If circuit is broken (we hit a 429), reduce refill rate permanently
        rate = self._rate_per_sec * 0.5 if self._circuit_broken else self._rate_per_sec
        
        self._tokens = min(self._capacity, self._tokens + elapsed * rate)

    async def _wait_tokens(self, cost: int, lane: str) -> None:
        """
        Wait until enough tokens are available.
        For non-exit lanes, keep cfg.exit_token_reserve tokens unspent.
        """
        if cost <= 0:
            return

        reserve = int(self.cfg.exit_token_reserve) if lane != "exit" else 0

        while True:
            async with self._token_lock:
                self._refill_locked()

                available = self._tokens
                effective_available = available - reserve

                if effective_available >= cost:
                    self._tokens -= cost
                    self.total_consumed += cost
                    return

                # compute wait time until enough tokens for this lane
                needed = cost - max(0.0, effective_available)
                
                rate = self._rate_per_sec * 0.5 if self._circuit_broken else self._rate_per_sec
                wait_time_raw = needed / max(0.0001, rate)

            # Wait outside lock
            wait_sec = max(0.01, wait_time_raw + 0.01)
            self.total_wait_sec += wait_sec
            await asyncio.sleep(wait_sec)

    async def call(
        self,
        *,
        cost: int,
        lane: str,
        fn: Callable[[], Awaitable[Any]],
        timeout_sec: Optional[float] = None,
    ) -> Any:
        """
        Execute a governed call.

        Args:
          cost: token cost (API weight unit).
          lane: "exit" | "pos" | "entry" | "scan" | "universe" | ...
          fn: a 0-arg coroutine factory.
          timeout_sec: override call timeout.
        """
        # CRITICAL: Log entry lane calls (orders) for debugging
        if lane == "entry":
            # Verbose logging disabled - too spammy
            pass
        
        # 1. Wait for tokens
        await self._wait_tokens(cost=cost, lane=lane)
        
        # CRITICAL: Log when order is about to execute
        if lane == "entry":
            # Verbose logging disabled - too spammy
            pass
        
        # 2. Execute with circuit breaker logic
        async with self._inflight:
            t = self.cfg.call_timeout_sec if timeout_sec is None else timeout_sec
            try:
                if t and t > 0:
                    result = await asyncio.wait_for(fn(), timeout=t)
                    # Log successful execution for entry lane
                    if lane == "entry":
                        # Verbose logging disabled - too spammy
                        pass
                    return result
                result = await fn()
                if lane == "entry":
                    # Verbose logging disabled - too spammy
                    pass
                return result
            except asyncio.TimeoutError:
                # CRITICAL: Log timeout for entry orders
                if lane == "entry":
                    try:
                        import sys
                        sys.stderr.write(f"[API_GOV] Entry order TIMEOUT after {t}s\n")
                    except Exception:
                        pass
                raise
            except Exception as e:
                # 3. Check for Rate Limit hits (429 / 418)
                err_str = str(e).lower()
                if "429" in err_str or "too many requests" in err_str or "418" in err_str:
                    if not self._circuit_broken:
                        self._circuit_broken = True
                        print("\n[CRITICAL] API RATE LIMIT HIT (429). ENGAGING CIRCUIT BREAKER. REDUCING SPEED BY 50%.\n")
                    # Force a penalty sleep to clear the ban window
                    await asyncio.sleep(5.0)
                # CRITICAL: Log exception for entry orders
                if lane == "entry":
                    try:
                        import sys
                        sys.stderr.write(f"[API_GOV] Entry order EXCEPTION: {type(e).__name__}: {str(e)}\n")
                    except Exception:
                        pass
                raise e
