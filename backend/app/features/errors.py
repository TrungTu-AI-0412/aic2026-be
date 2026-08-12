class FeatureExtractionError(Exception):
    """Raised when media or a model cannot produce a valid feature vector."""


class UnknownFeatureProfileError(Exception):
    """Raised when a job asks for an unregistered embedding profile."""
