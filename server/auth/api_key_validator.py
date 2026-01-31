"""
API Key Validator

Validates API keys for programmatic access from AR headsets.
"""

import hashlib
import secrets
from typing import Optional, Set
from loguru import logger

from ..config import get_config


class APIKeyValidator:
    """Validates API keys for client authentication."""

    def __init__(self):
        """Initialize API key validator."""
        self.config = get_config()
        # In production, these would be loaded from database/secrets manager
        self._valid_keys: Set[str] = set()
        self._load_keys()

    def _load_keys(self):
        """Load API keys from configuration or secrets manager."""
        # For demo purposes, generate a default key
        # In production, load from AWS Secrets Manager or database
        default_key = self._generate_api_key()
        self._valid_keys.add(self._hash_key(default_key))

        logger.info(f"Loaded {len(self._valid_keys)} API keys")
        logger.warning(f"Demo API Key (save this!): {default_key}")

    def _generate_api_key(self, prefix: str = "sar") -> str:
        """
        Generate a new API key.

        Args:
            prefix: Key prefix for identification.

        Returns:
            API key string.
        """
        # Generate 32 random bytes -> 64 character hex string
        random_part = secrets.token_hex(32)
        return f"{prefix}_{random_part}"

    def _hash_key(self, api_key: str) -> str:
        """
        Hash an API key for secure storage.

        Args:
            api_key: Plain API key.

        Returns:
            Hashed key.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.

        Args:
            api_key: API key to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not api_key:
            return False

        hashed = self._hash_key(api_key)
        is_valid = hashed in self._valid_keys

        if not is_valid:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")

        return is_valid

    def add_key(self, api_key: str) -> bool:
        """
        Add a new API key to the valid set.

        Args:
            api_key: API key to add.

        Returns:
            True if added, False if already exists.
        """
        hashed = self._hash_key(api_key)

        if hashed in self._valid_keys:
            return False

        self._valid_keys.add(hashed)
        logger.info("Added new API key")
        return True

    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: API key to revoke.

        Returns:
            True if revoked, False if not found.
        """
        hashed = self._hash_key(api_key)

        if hashed not in self._valid_keys:
            return False

        self._valid_keys.remove(hashed)
        logger.warning(f"Revoked API key: {api_key[:10]}...")
        return True

    def create_new_key(self) -> str:
        """
        Create and register a new API key.

        Returns:
            New API key string.
        """
        new_key = self._generate_api_key()
        self.add_key(new_key)
        return new_key


# Global API key validator instance
_api_key_validator: Optional[APIKeyValidator] = None


def get_api_key_validator() -> APIKeyValidator:
    """Get the global API key validator instance."""
    global _api_key_validator
    if _api_key_validator is None:
        _api_key_validator = APIKeyValidator()
    return _api_key_validator
