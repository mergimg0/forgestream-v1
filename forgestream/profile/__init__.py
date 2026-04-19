"""User profile extraction — Theme 5."""

from .model import UserProfile
from .extractor import UserProfileExtractor
from .adaptation import StyleAdapter

__all__ = ["UserProfile", "UserProfileExtractor", "StyleAdapter"]
