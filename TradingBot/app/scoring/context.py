"""Signal Context - Input object for scoring."""

from dataclasses import dataclass
from typing import Any, Literal, Dict, Optional


Side = Literal["long", "short"]


@dataclass
class SignalContext:
    """
    Minimal context required for scoring.
    Extend as needed; keep this as the only dependency boundary.
    """
    symbol: str
    side: Side
    base_score: float   # from your existing local/indicator signal logic (0–100)
    now_ts: int         # current timestamp (e.g. seconds since epoch)
    hour_utc: int       # for time-of-day scoring (0–23)
    session: str        # "ASIA", "EU", "US", "WEEKEND", etc.

    # Market microstructure
    spread_pct: float             # (ask-bid)/mid
    depth_top_usd: float          # top-of-book depth within small price band
    atr_pct: float                # ATR / price on LTF
    htf_adx: float                # 1h ADX(14)

    # Structure / pattern flags
    htf_trend_dir: int           # 1, 0, -1
    at_range_high: bool
    at_range_low: bool
    extreme_up: bool             # overextension flags
    extreme_down: bool
    bearish_divergence: bool
    bullish_divergence: bool
    exhaustion_up: bool
    exhaustion_down: bool
    sfp_top: bool
    sfp_bottom: bool
    dist_from_vwap: float        # signed distance fraction

    # Portfolio and symbol stats
    open_positions_same_sector: int
    corr_to_btc_24h: float
    symbol_rating: float         # from historical perf [-10,+10] pre-clamped

    extra: Optional[Dict[str, Any]] = None  # extension hook

