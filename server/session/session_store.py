"""
Session Store

Manages active sessions with DynamoDB persistence (in production) or
in-memory storage (for development).
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from loguru import logger

from ..config import get_config


@dataclass
class Session:
    """Represents an active client session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"  # active, paused, terminated
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_heartbeat) > timeout_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SessionStore:
    """
    Session storage with optional DynamoDB persistence.

    In development mode, uses in-memory storage.
    In production, integrates with DynamoDB for persistence across instances.
    """

    def __init__(self, use_dynamodb: bool = False):
        """
        Initialize session store.

        Args:
            use_dynamodb: If True, use DynamoDB for persistence.
        """
        self.config = get_config()
        self.use_dynamodb = use_dynamodb
        self._sessions: Dict[str, Session] = {}

        if use_dynamodb:
            self._init_dynamodb()
        else:
            logger.info("Using in-memory session store (development mode)")

    def _init_dynamodb(self):
        """Initialize DynamoDB client."""
        try:
            import boto3

            self.dynamodb = boto3.resource("dynamodb", region_name=self.config.AWS_REGION)
            self.table = self.dynamodb.Table(self.config.DYNAMODB_TABLE_SESSIONS)

            logger.success("DynamoDB session store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB: {e}")
            logger.warning("Falling back to in-memory store")
            self.use_dynamodb = False

    async def create_session(
        self, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier.
            metadata: Optional session metadata.

        Returns:
            Created Session object.
        """
        session = Session(session_id=session_id, metadata=metadata or {})

        self._sessions[session_id] = session

        if self.use_dynamodb:
            await self._save_to_dynamodb(session)

        logger.info(f"Created session: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session object or None if not found.
        """
        # Check in-memory first
        session = self._sessions.get(session_id)

        if not session and self.use_dynamodb:
            # Try loading from DynamoDB
            session = await self._load_from_dynamodb(session_id)
            if session:
                self._sessions[session_id] = session

        return session

    async def update_heartbeat(self, session_id: str):
        """
        Update session heartbeat timestamp.

        Args:
            session_id: Session identifier.
        """
        session = await self.get_session(session_id)

        if session:
            session.last_heartbeat = time.time()

            if self.use_dynamodb:
                await self._save_to_dynamodb(session)

    async def update_status(self, session_id: str, status: str):
        """
        Update session status.

        Args:
            session_id: Session identifier.
            status: New status.
        """
        session = await self.get_session(session_id)

        if session:
            session.status = status

            if self.use_dynamodb:
                await self._save_to_dynamodb(session)

            logger.info(f"Session {session_id} status changed to: {status}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted, False if not found.
        """
        session = self._sessions.pop(session_id, None)

        if not session:
            return False

        if self.use_dynamodb:
            await self._delete_from_dynamodb(session_id)

        logger.info(f"Deleted session: {session_id}")
        return True

    async def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        timeout = self.config.SESSION_TIMEOUT_SECONDS
        expired_sessions = []

        for session_id, session in self._sessions.items():
            if session.is_expired(timeout):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.delete_session(session_id)
            logger.info(f"Removed expired session: {session_id}")

    async def get_all_sessions(self) -> Dict[str, Session]:
        """Get all active sessions."""
        return self._sessions.copy()

    async def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    # DynamoDB operations (async wrappers)

    async def _save_to_dynamodb(self, session: Session):
        """Save session to DynamoDB."""
        if not self.use_dynamodb:
            return

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.table.put_item(
                    Item={
                        "session_id": session.session_id,
                        "created_at": session.created_at,
                        "last_heartbeat": session.last_heartbeat,
                        "status": session.status,
                        "metadata": session.metadata,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Failed to save session to DynamoDB: {e}")

    async def _load_from_dynamodb(self, session_id: str) -> Optional[Session]:
        """Load session from DynamoDB."""
        if not self.use_dynamodb:
            return None

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.table.get_item(Key={"session_id": session_id})
            )

            item = response.get("Item")
            if item:
                return Session(
                    session_id=item["session_id"],
                    created_at=item["created_at"],
                    last_heartbeat=item["last_heartbeat"],
                    status=item["status"],
                    metadata=item.get("metadata", {}),
                )

            return None

        except Exception as e:
            logger.error(f"Failed to load session from DynamoDB: {e}")
            return None

    async def _delete_from_dynamodb(self, session_id: str):
        """Delete session from DynamoDB."""
        if not self.use_dynamodb:
            return

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.table.delete_item(Key={"session_id": session_id})
            )
        except Exception as e:
            logger.error(f"Failed to delete session from DynamoDB: {e}")


# Global session store instance
_session_store: Optional[SessionStore] = None


def get_session_store(use_dynamodb: bool = False) -> SessionStore:
    """Get the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(use_dynamodb=use_dynamodb)
    return _session_store
