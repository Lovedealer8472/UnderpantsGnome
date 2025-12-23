"""
Global Safety Wrapper - Provides clean exception tracing and error handling.
Wraps critical operations to ensure clean error reporting without disrupting execution.
"""

import functools
import traceback
from typing import Callable, TypeVar, Any, Optional
from .logger import get_logger

T = TypeVar('T')
logger = get_logger("SafetyWrapper")


def safe_execute(
    operation_name: str,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False
) -> Callable:
    """
    Decorator for safe execution with clean exception tracing.
    
    Args:
        operation_name: Name of the operation for logging
        default_return: Value to return on exception (if not reraise)
        log_level: Log level ("debug", "info", "warning", "error", "critical")
        reraise: If True, re-raise exception after logging
    
    Returns:
        Decorated function
    
    Example:
        @safe_execute("fetch_ticker", default_return=None)
        async def fetch_ticker(symbol: str):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{operation_name} failed: {type(e).__name__}: {str(e)}"
                error_context = {
                    'operation': operation_name,
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                
                # Add args/kwargs context if debug mode
                if logger.isEnabledFor(10):  # DEBUG level
                    error_context['args'] = str(args)[:200]  # Limit length
                    error_context['kwargs'] = str(kwargs)[:200]
                
                # Log with appropriate level
                log_method = getattr(logger, log_level, logger.error)
                log_method(error_msg, **error_context)
                
                # Log full traceback to file (always)
                logger.logger.debug(
                    f"Full traceback for {operation_name}:\n{traceback.format_exc()}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{operation_name} failed: {type(e).__name__}: {str(e)}"
                error_context = {
                    'operation': operation_name,
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                
                # Add args/kwargs context if debug mode
                if logger.isEnabledFor(10):  # DEBUG level
                    error_context['args'] = str(args)[:200]
                    error_context['kwargs'] = str(kwargs)[:200]
                
                # Log with appropriate level
                log_method = getattr(logger, log_level, logger.error)
                log_method(error_msg, **error_context)
                
                # Log full traceback to file (always)
                logger.logger.debug(
                    f"Full traceback for {operation_name}:\n{traceback.format_exc()}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                return default_return
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default_return: Any = None,
    operation_name: Optional[str] = None,
    log_level: str = "error",
    **kwargs: Any
) -> T:
    """
    Safely call a function with exception handling.
    
    Args:
        func: Function to call
        *args: Positional arguments
        default_return: Value to return on exception
        operation_name: Name of operation for logging
        log_level: Log level
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on exception
    
    Example:
        result = safe_call(risky_function, arg1, arg2, default_return=None)
    """
    op_name = operation_name or func.__name__
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = {
            'operation': op_name,
            'function': func.__name__,
            'error_type': type(e).__name__,
            'error_message': str(e)
        }
        
        log_method = getattr(logger, log_level, logger.error)
        log_method(f"{op_name} failed: {type(e).__name__}: {str(e)}", **error_context)
        
        logger.logger.debug(
            f"Full traceback for {op_name}:\n{traceback.format_exc()}",
            exc_info=True
        )
        
        return default_return


async def safe_await(
    coro: Any,
    default_return: Any = None,
    operation_name: Optional[str] = None,
    log_level: str = "error"
) -> Any:
    """
    Safely await a coroutine with exception handling.
    
    Args:
        coro: Coroutine to await
        default_return: Value to return on exception
        operation_name: Name of operation for logging
        log_level: Log level
    
    Returns:
        Coroutine result or default_return on exception
    
    Example:
        result = await safe_await(risky_coroutine(), default_return=None)
    """
    op_name = operation_name or "async_operation"
    try:
        return await coro
    except Exception as e:
        error_context = {
            'operation': op_name,
            'error_type': type(e).__name__,
            'error_message': str(e)
        }
        
        log_method = getattr(logger, log_level, logger.error)
        log_method(f"{op_name} failed: {type(e).__name__}: {str(e)}", **error_context)
        
        logger.logger.debug(
            f"Full traceback for {op_name}:\n{traceback.format_exc()}",
            exc_info=True
        )
        
        return default_return


