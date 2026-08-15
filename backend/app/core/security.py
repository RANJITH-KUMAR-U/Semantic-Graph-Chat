"""
Auth / session security placeholder.

Pre-launch hardening items to implement here:
  - API key validation middleware
  - Session token generation and verification
  - Rate limiting per session_id

For the MVP prototype, all endpoints are open — no auth required.
"""


def get_current_session_id(session_id: str) -> str:
    """
    Placeholder: validate and return the session ID.

    In production, this would verify a bearer token or signed session
    cookie and extract the session_id from it.
    """
    # TODO: add JWT/API-key validation before production deployment
    return session_id
