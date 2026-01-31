"""
Tests for authentication components
"""

import pytest
from server.auth.jwt_manager import JWTManager
from server.auth.api_key_validator import APIKeyValidator


def test_jwt_token_creation():
    """Test JWT token creation."""
    jwt_manager = JWTManager()

    token = jwt_manager.create_access_token("test_user")
    assert token is not None
    assert isinstance(token, str)


def test_jwt_token_verification():
    """Test JWT token verification."""
    jwt_manager = JWTManager()

    # Create token
    token = jwt_manager.create_access_token("test_user", {"role": "admin"})

    # Verify token
    payload = jwt_manager.verify_token(token, token_type="access")

    assert payload is not None
    assert payload["sub"] == "test_user"
    assert payload["role"] == "admin"


def test_jwt_invalid_token():
    """Test handling of invalid JWT token."""
    jwt_manager = JWTManager()

    payload = jwt_manager.verify_token("invalid_token")
    assert payload is None


def test_refresh_token_flow():
    """Test refresh token workflow."""
    jwt_manager = JWTManager()

    # Create refresh token
    refresh_token = jwt_manager.create_refresh_token("test_user")

    # Use refresh token to get new access token
    new_access_token = jwt_manager.refresh_access_token(refresh_token)

    assert new_access_token is not None

    # Verify new access token
    payload = jwt_manager.verify_token(new_access_token)
    assert payload["sub"] == "test_user"


def test_api_key_validation():
    """Test API key validation."""
    validator = APIKeyValidator()

    # Create a new key
    api_key = validator.create_new_key()
    assert api_key.startswith("sar_")

    # Validate key
    assert validator.validate_key(api_key)

    # Invalid key
    assert not validator.validate_key("invalid_key")


def test_api_key_revocation():
    """Test API key revocation."""
    validator = APIKeyValidator()

    # Create key
    api_key = validator.create_new_key()
    assert validator.validate_key(api_key)

    # Revoke key
    assert validator.revoke_key(api_key)

    # Should no longer be valid
    assert not validator.validate_key(api_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
