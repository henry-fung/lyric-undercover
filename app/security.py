import hashlib
import secrets


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_recovery_code() -> str:
    # Short enough to type, but generated with cryptographic randomness.
    return secrets.token_hex(5).upper()
