"""
Helper functions for setting up logging in Jupyter notebooks
"""

import logging
import sys


def setup_notebook_logging(level=logging.DEBUG, 
                          format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                          date_format='%Y-%m-%d %H:%M:%S'):
    """
    Set up logging for Jupyter notebooks with proper configuration.
    
    Args:
        level: Logging level (default: logging.DEBUG)
        format_string: Format string for log messages
        date_format: Date format for timestamps
    
    Returns:
        The root logger instance
    """
    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with force=True to override any existing configuration
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        force=True,  # This is important in Jupyter to reconfigure logging
        handlers=[
            logging.StreamHandler(sys.stdout)  # Explicitly use stdout
        ]
    )
    
    # Return the logger for additional configuration if needed
    return logging.getLogger()


def get_module_logger(module_name, level=None):
    """
    Get a logger for a specific module with optional level override.
    
    Args:
        module_name: Name of the module (e.g., 'mkt_data_yfinance')
        level: Optional logging level for this specific logger
    
    Returns:
        Logger instance for the module
    """
    logger = logging.getLogger(module_name)
    if level is not None:
        logger.setLevel(level)
    return logger


def setup_file_logging(filename='trading_debug.log', 
                      level=logging.DEBUG,
                      format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      date_format='%Y-%m-%d %H:%M:%S'):
    """
    Add file logging in addition to console output.
    
    Args:
        filename: Log file name
        level: Logging level for file output
        format_string: Format string for log messages
        date_format: Date format for timestamps
    """
    # Create file handler
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)
    
    return file_handler


def disable_external_loggers():
    """
    Disable or reduce logging from external libraries that might be too verbose.
    """
    # Common noisy loggers
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


# Convenience function for quick setup
def quick_setup(debug=True):
    """
    Quick setup for common use case.
    
    Args:
        debug: If True, set level to DEBUG; if False, set to INFO
    """
    level = logging.DEBUG if debug else logging.INFO
    setup_notebook_logging(level=level)
    disable_external_loggers()
    print(f"Logging configured at {logging.getLevelName(level)} level") 