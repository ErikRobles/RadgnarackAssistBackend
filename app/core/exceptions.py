class RadgnarackError(Exception):
    """Base error for the application."""
    pass

class ConfigurationError(RadgnarackError):
    """Raised when application configuration is missing or invalid."""
    pass

class RetrievalError(RadgnarackError):
    """Raised when the retrieval process fails."""
    pass

class ServiceUnavailableError(RadgnarackError):
    """Raised when an external service is unreachable."""
    pass
