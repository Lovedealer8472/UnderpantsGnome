"""
SIMPLE STOP LOSS MANAGER
========================

Single source of truth for ALL stop loss logic.
- Set SL at entry (1.5% from entry)
- Trail SL when in profit (0.3% from current)
- Check SL hit on price update

No complexity. No grace periods. No conflicts.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SimpleStopLossManager:
    """Single, simple, correct stop loss manager."""
    
    # Configuration - CRYPTO-APPROPRIATE
    MIN_INITIAL_SL_PCT = 0.025      # 2.5% below entry
    ATR_MULTIPLIER = 4.0            # 4.0x ATR for volatility
    MAX_SL_PCT = 0.15               # Max 15% stop distance

    # TRAILING STOPS - REPLACED with profit-based trailing
    # OLD: 0.5% below current price (TOO TIGHT for crypto!)
    # NEW: Trail from peak profit level, not current price
    TRAIL_FROM_PEAK_PCT = 0.03     # 3% below peak profit (gives room for pullbacks)
    TRAIL_ACTIVATION_PCT = 0.015    # Activate trailing after 1.5% profit
    
    @staticmethod
    def set_initial_stop_loss(entry_price: float, side: str, atr_pct: float = None, symbol: str = None) -> float:
        """
        Calculate INITIAL stop loss for a new entry.
        
        This is the protective stop - stays fixed until we're in profit.
        
        Rules:
        - LONG: SL is BELOW entry price
        - SHORT: SL is ABOVE entry price
        - Minimum: 1.5% from entry
        - Maximum: 10% from entry
        - Use 2.5x ATR if available (whichever is LARGER)
        
        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr_pct: Optional ATR as decimal (0.01 = 1%)
        
        Returns:
            Stop loss price
        """
        if entry_price <= 0:
            logger.error(f"[SL] Invalid entry price: {entry_price}")
            return 0
        
        # Calculate stop distance
        if atr_pct and atr_pct > 0:
            # Base ATR multiplier
            multiplier = SimpleStopLossManager.ATR_MULTIPLIER

            # ML-LEARNED SYMBOL-SPECIFIC ADJUSTMENTS
            # Use symbol profiles from Master Hindsight ML instead of hardcoded rules
            if symbol:
                try:
                    from pathlib import Path
                    import json

                    # Load latest symbol profiles
                    profiles_dir = Path("master_hindsight_models")
                    if profiles_dir.exists():
                        model_dirs = sorted(profiles_dir.glob("*"), reverse=True)
                        if model_dirs:
                            profile_file = model_dirs[0] / "symbol_profiles.json"
                            if profile_file.exists():
                                with open(profile_file, 'r') as f:
                                    symbol_profiles = json.load(f)

                                symbol_upper = symbol.upper()
                                if symbol_upper in symbol_profiles:
                                    profile = symbol_profiles[symbol_upper]
                                    optimal_sl = profile.get('optimal_sl', SimpleStopLossManager.ATR_MULTIPLIER)

                                    # Use ML-learned optimal SL, but never more aggressive than our base
                                    # (ML might suggest tighter stops for well-behaved symbols)
                                    multiplier = max(optimal_sl, SimpleStopLossManager.ATR_MULTIPLIER)

                                    logger.debug(
                                        f"[SL_ML] {symbol}: Using ML-learned multiplier {multiplier:.1f}x "
                                        f"(profile: WR {profile.get('win_rate',0)*100:.1f}%, "
                                        f"avg_R {profile.get('avg_r',0):.1f})"
                                    )
                except Exception as e:
                    logger.debug(f"[SL_ML] Failed to load symbol profile for {symbol}: {e}")
                    # Fall back to standard logic

            atr_distance = multiplier * atr_pct
            # But never less than minimum
            stop_distance = max(atr_distance, SimpleStopLossManager.MIN_INITIAL_SL_PCT)
        else:
            # No ATR? Use minimum
            stop_distance = SimpleStopLossManager.MIN_INITIAL_SL_PCT
        
        # Clamp between min and max
        stop_distance = max(
            SimpleStopLossManager.MIN_INITIAL_SL_PCT,
            min(stop_distance, SimpleStopLossManager.MAX_SL_PCT)
        )
        
        # Apply to entry price
        if side.lower() == 'long':
            sl = entry_price * (1.0 - stop_distance)
        else:  # short
            sl = entry_price * (1.0 + stop_distance)
        
        logger.info(
            f"[SL_SET] {side.upper()}: Entry=${entry_price:.2f}, "
            f"SL=${sl:.2f} ({stop_distance*100:.1f}% from entry)"
        )
        
        return sl
    
    @staticmethod
    def check_sl_hit(current_price: float, stop_loss: float, side: str) -> bool:
        """
        Check if stop loss has been hit.
        
        Args:
            current_price: Current market price
            stop_loss: Stop loss price
            side: 'long' or 'short'
        
        Returns:
            True if price has hit or crossed SL
        """
        if current_price <= 0 or stop_loss <= 0:
            return False
        
        if side.lower() == 'long':
            # LONG: SL hit if current_price drops to or below SL
            return current_price <= stop_loss
        else:  # short
            # SHORT: SL hit if current_price rises to or above SL
            return current_price >= stop_loss
    
    @staticmethod
    def update_trailing_stop(
        entry_price: float,
        current_price: float,
        current_sl: float,
        side: str
    ) -> Tuple[float, bool, float]:
        """
        Update stop loss while in profit.
        
        CRYPTO-APPROPRIATE PROFIT-BASED TRAILING:
        - Trail 3% below PEAK PROFIT (not current price)
        - Only activate after 1.5% profit
        - Gives room for normal crypto pullbacks
        - Return updated SL and whether it changed
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_sl: Current stop loss price
            side: 'long' or 'short'
        
        Returns:
            (new_sl_price, did_update, profit_locked_pct)
        """
        if side.lower() == 'long':
            # LONG position
            profit = current_price - entry_price
            profit_pct = (profit / entry_price) * 100 if entry_price > 0 else 0
            
            # Only trail if in profit AND above activation threshold (1.5% profit)
            if current_price <= entry_price or profit_pct < SimpleStopLossManager.TRAIL_ACTIVATION_PCT * 100:
                return current_sl, False, 0

            # CRYPTO SOLUTION: Trail 3% below ENTRY PRICE (not current price!)
            # This gives the trade room for pullbacks while protecting profits
            trailing_sl = entry_price * (1.0 + profit_pct/100 - SimpleStopLossManager.TRAIL_FROM_PEAK_PCT)
            
            # Only move SL UP (tighten), never down (loosen)
            if trailing_sl > current_sl:
                # Calculate profit now locked by new SL
                profit_locked = (trailing_sl - entry_price) / entry_price * 100
                
                logger.info(
                    f"[SL_TRAIL] LONG: ${current_sl:.2f} -> ${trailing_sl:.2f} | "
                    f"Entry: ${entry_price:.2f} | Current profit: {profit_pct:.2f}% | "
                    f"Profit locked: {profit_locked:.2f}% (3% trail from entry + profit)"
                )
                
                return trailing_sl, True, profit_locked
            else:
                return current_sl, False, 0
        
        else:  # SHORT
            # SHORT position
            profit = entry_price - current_price
            profit_pct = (profit / entry_price) * 100 if entry_price > 0 else 0
            
            # Only trail if in profit AND above activation threshold (1.5% profit)
            if current_price >= entry_price or profit_pct < SimpleStopLossManager.TRAIL_ACTIVATION_PCT * 100:
                return current_sl, False, 0

            # CRYPTO SOLUTION: Trail 3% above ENTRY PRICE (not current price!)
            trailing_sl = entry_price * (1.0 - profit_pct/100 + SimpleStopLossManager.TRAIL_FROM_PEAK_PCT)
            
            # Only move SL DOWN (tighten), never up (loosen)
            if trailing_sl < current_sl:
                # Calculate profit now locked by new SL
                profit_locked = (entry_price - trailing_sl) / entry_price * 100
                
                logger.info(
                    f"[SL_TRAIL] SHORT: ${current_sl:.2f} -> ${trailing_sl:.2f} | "
                    f"Entry: ${entry_price:.2f} | Current profit: {profit_pct:.2f}% | "
                    f"Profit locked: {profit_locked:.2f}% (3% trail from entry + profit)"
                )
                
                return trailing_sl, True, profit_locked
            else:
                return current_sl, False, 0
    
    @staticmethod
    def validate_sl(entry_price: float, stop_loss: float, side: str, current_price: float = None) -> Tuple[bool, str]:
        """
        Validate that stop loss is in correct position.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'long' or 'short'
            current_price: Optional current price (for trailing stop validation)
        
        Returns:
            (is_valid, reason)
        """
        if entry_price <= 0:
            return False, "invalid_entry_price"
        
        if stop_loss <= 0:
            return False, "stop_loss_zero_or_negative"
        
        if side.lower() == 'long':
            # LONG: SL should be below entry (or possibly above entry if trailing in profit)
            if stop_loss >= entry_price:
                # Exception: trailing stop above entry (valid if in profit)
                if current_price and current_price > entry_price and stop_loss < current_price:
                    return True, "valid_trailing_stop_above_entry"
                return False, "long_sl_above_entry"
            return True, "valid_long_sl"
        
        elif side.lower() == 'short':
            # SHORT: SL should be above entry (or possibly below entry if trailing in profit)
            if stop_loss <= entry_price:
                # Exception: trailing stop below entry (valid if in profit)
                if current_price and current_price < entry_price and stop_loss > current_price:
                    return True, "valid_trailing_stop_below_entry"
                return False, "short_sl_below_entry"
            return True, "valid_short_sl"
        
        return False, "invalid_side"


# Usage example
if __name__ == "__main__":
    print("[TEST] Simple Stop Loss Manager")
    print("=" * 70)
    
    # Test 1: LONG initial SL
    print("\n1. LONG Entry at $100 (ATR=0.5%)")
    sl = SimpleStopLossManager.set_initial_stop_loss(100, 'long', 0.005)
    print(f"   Entry: $100, SL: ${sl:.2f}")
    
    # Test 2: SHORT initial SL
    print("\n2. SHORT Entry at $100 (ATR=0.5%)")
    sl = SimpleStopLossManager.set_initial_stop_loss(100, 'short', 0.005)
    print(f"   Entry: $100, SL: ${sl:.2f}")
    
    # Test 3: Check SL hit
    print("\n3. Check SL Hit")
    print(f"   LONG: Current=$98.50, SL=$98.50 -> HIT={SimpleStopLossManager.check_sl_hit(98.50, 98.50, 'long')}")
    print(f"   LONG: Current=$98.40, SL=$98.50 -> HIT={SimpleStopLossManager.check_sl_hit(98.40, 98.50, 'long')}")
    
    # Test 4: Trail in profit
    print("\n4. Trail SL in Profit")
    new_sl, updated, profit = SimpleStopLossManager.update_trailing_stop(100, 105, 98.5, 'long')
    print(f"   Entry=$100, Current=$105 (5% profit), Old SL=$98.50")
    print(f"   New SL=${new_sl:.2f}, Updated={updated}, Profit locked={profit:.2f}%")
    
    # Test 5: Validate SL
    print("\n5. Validate SL")
    valid, reason = SimpleStopLossManager.validate_sl(100, 98.5, 'long')
    print(f"   Entry=$100, SL=$98.50, LONG -> {valid} ({reason})")

