"""
JWT Token Manager

Handles JWT token generation, validation, and refresh for secure API access.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from loguru import logger

from ..config import get_config


class JWTManager:
    """Manager for JWT token operations."""

    def __init__(self):
        """Initialize JWT manager with config."""
        self.config = get_config()
        self.secret_key = self.config.JWT_SECRET_KEY
        self.algorithm = self.config.JWT_ALGORITHM
        self.access_token_expire_minutes = self.config.JWT_EXPIRATION_MINUTES
        self.refresh_token_expire_days = self.config.JWT_REFRESH_DAYS

    def create_access_token(
        self, subject: str, additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            subject: Token subject (typically user/session ID).
            additional_claims: Additional claims to include in token.

        Returns:
            Encoded JWT token string.
        """
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        expire = datetime.utcnow() + expires_delta

        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        if additional_claims:
            claims.update(additional_claims)

        try:
            encoded_jwt = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Created access token for subject: {subject}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise

    def create_refresh_token(self, subject: str) -> str:
        """
        Create a JWT refresh token.

        Args:
            subject: Token subject (typically user/session ID).

        Returns:
            Encoded JWT refresh token string.
        """
        expires_delta = timedelta(days=self.refresh_token_expire_days)
        expire = datetime.utcnow() + expires_delta

        claims = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        try:
            encoded_jwt = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"Created refresh token for subject: {subject}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise

    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string.
            token_type: Expected token type ('access' or 'refresh').

        Returns:
            Decoded token payload or None if invalid.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != token_type:
                logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create a new access token from a refresh token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            New access token or None if refresh token is invalid.
        """
        payload = self.verify_token(refresh_token, token_type="refresh")

        if not payload:
            return None

        subject = payload.get("sub")
        if not subject:
            return None

        # Create new access token
        return self.create_access_token(subject)

    def get_token_subject(self, token: str) -> Optional[str]:
        """
        Extract subject from token without full verification.

        Args:
            token: JWT token string.

        Returns:
            Token subject or None.
        """
        try:
            # Decode without verification to get subject
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload.get("sub")
        except Exception:
            return None


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager
