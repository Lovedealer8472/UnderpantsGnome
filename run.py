"""
Trading bot entry point.

CRITICAL: NO console output when UI v2 is active.
All logs/errors go to file logger.
"""

import asyncio
import sys
import signal
import os
import logging
from pathlib import Path

# Fix Windows console encoding for Unicode/emoji characters
if os.name == 'nt':
    try:
        import io
        # reconfigure to utf-8 and line buffering
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
            sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
        else:
            # Fallback for older python
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass  # Fallback if encoding can't be changed

# Load .env file from config folder or root
try:
    from dotenv import load_dotenv
    # Try config/.env first, then fall back to root .env
    config_env = Path("config/.env")
    root_env = Path(".env")
    if config_env.exists():
        load_dotenv(config_env)
    elif root_env.exists():
        load_dotenv(root_env)
except ImportError:
    pass  # dotenv is optional

from app.bot import ScalperBot
from app.config_validator import validate_and_exit_on_error
from app.logger import get_logger


def check_single_instance(lock_file: Path = Path("run.lock")):
    """
    Check if another instance is already running.
    
    Returns:
        (is_running, message) - True if another instance is running, False otherwise
    """
    if not lock_file.exists():
        return False, ""
    
    try:
        # Read PID from lock file
        with open(lock_file, 'r') as f:
            pid_str = f.read().strip()
        
        if not pid_str:
            # Empty lock file - stale, can be removed
            return False, "stale_empty"
        
        try:
            pid = int(pid_str)
        except ValueError:
            # Invalid PID - stale, can be removed
            return False, "stale_invalid"
        
        # Check if process is still running
        if os.name == 'nt':
            # Windows: Use tasklist or check process existence
            try:
                import psutil
                if psutil.pid_exists(pid):
                    return True, f"PID {pid} is running"
            except ImportError:
                # Fallback: Try to send signal 0 (doesn't work on Windows, but try anyway)
                try:
                    os.kill(pid, 0)
                    return True, f"PID {pid} is running"
                except (OSError, ProcessLookupError):
                    # Process doesn't exist
                    return False, "stale_pid"
        else:
            # Unix: Send signal 0 to check if process exists
            try:
                os.kill(pid, 0)
                return True, f"PID {pid} is running"
            except (OSError, ProcessLookupError):
                # Process doesn't exist
                return False, "stale_pid"
    
    except Exception as e:
        # Error reading lock file - assume stale
        return False, f"stale_error: {e}"
    
    return False, "stale_unknown"


def create_lock_file(lock_file: Path = Path("run.lock")) -> bool:
    """Create lock file with current PID. Returns True on success."""
    try:
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception:
        return False


def remove_lock_file(lock_file: Path = Path("run.lock")):
    """Remove lock file. Safe to call even if file doesn't exist."""
    try:
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass  # Ignore errors during cleanup


async def main():
    """
    Main entry point for the trading bot.
    All output goes to file logger, NOT console (to preserve UI).
    """
    # Lock file path (kept for cleanup only, lock file creation is disabled)
    lock_file = Path("run.lock")
    logger = None
    bot = None
    
    # STARTUP DIAGNOSTICS (console output OK before UI starts, but route to logger for consistency)
    # Get logger early (before UI starts) - console output allowed here
    try:
        logger = get_logger("TradingBot", enable_console=True)  # Enable console ONLY for startup
    except Exception:
        # Fallback: use basic logging if logger init fails
        import logging as py_logging
        py_logging.basicConfig(level=py_logging.INFO, format='%(message)s')
        logger = py_logging.getLogger("TradingBot")
    
    logger.info("[STARTUP] Validating configuration...")
    try:
        validate_and_exit_on_error()
        logger.info("[STARTUP] OK Configuration validated")
    except Exception as e:
        logger.error(f"[STARTUP] FAIL Configuration validation failed: {e}")
        logger.exception("Configuration validation traceback:")
        sys.exit(1)
    
    logger.info("[STARTUP] OK Logger initialized")
    logger.info("="*80)
    logger.info("TRADING BOT STARTING")
    logger.info("="*80)
    
    logger.info("[STARTUP] Creating bot instance...")
    logger.info("[STARTUP] UI will take over in 3 seconds...")
    logger.info("[STARTUP] If bot crashes, check logs/bot_*.log for details")
    
    try:
        bot = ScalperBot()
        logger.info("[STARTUP] OK Bot instance created")
    except Exception as e:
        logger.error(f"[STARTUP] FAIL Bot instance creation failed: {e}")
        logger.exception("Bot instance creation traceback:")
        remove_lock_file(lock_file)
        sys.exit(1)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Only disable console if using Rich UI (which handles its own output)
    from app.config import USE_RICH_UI
    if USE_RICH_UI:
    # CRITICAL: Disable console output from logger NOW (before UI starts)
    # Re-initialize logger with console disabled
        logger = get_logger("TradingBot", enable_console=False)
    
    try:
        logger.info("[STARTUP] Starting bot.run()...")
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Restore stdout/stderr for safe logging
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        if logger:
            logger.info("="*80)
            logger.info("BOT SHUTDOWN")
            logger.info("="*80)
            logger.info("Manual shutdown requested (Ctrl+C)")
            logger.info("Saving state and closing connections...")
        
        # Save state
        try:
            if bot and hasattr(bot, 'fast_storage'):
                bot.fast_storage.close()
            if logger:
                logger.info("OK State saved")
        except Exception as e:
            if logger:
                logger.error(f"WARNING State save failed: {e}")
        
        # Close exchange
        try:
            if bot:
                if bot.exchange_wrapper:
                    await bot.exchange_wrapper.close()
                elif bot.exchange:
                    await bot.exchange.close()
            if logger:
                logger.info("OK Exchange connection closed")
        except Exception as e:
            if logger:
                logger.error(f"Exchange close failed: {e}")
        
        if logger:
            logger.info("Bot shutdown complete.")
            logger.info("="*80)
        
    except Exception as e:
        # Restore stdout/stderr for safe logging
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Log exception to file with full traceback
        if logger:
            logger.critical(f"FATAL ERROR: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
        
        # DO NOT print to console - only log to file
        raise  # Re-raise to be caught by outer handler
    
    finally:
        # CRITICAL: Clean up lock file if it exists (safety cleanup only, lock file creation is disabled)
        remove_lock_file(lock_file)
        
        # Shutdown logging to flush and close all handlers
        logging.shutdown()


if __name__ == "__main__":
    lock_file = Path("run.lock")
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Final fallback - log to file if possible
        try:
            logger = get_logger("TradingBot")
            logger.info("[SHUTDOWN] Bot stopped by user")
        except Exception:
            pass  # Silently exit
        finally:
            remove_lock_file(lock_file)
            logging.shutdown()
        sys.exit(0)
    except Exception as e:
        # Final fallback - log to file if possible
        try:
            logger = get_logger("TradingBot")
            logger.critical(f"[FATAL] Unhandled exception: {e}")
            logger.exception("Full traceback:")
        except Exception:
            pass  # Silently exit
        finally:
            remove_lock_file(lock_file)
            logging.shutdown()
        sys.exit(1)
