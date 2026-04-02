#!/usr/bin/env python3
"""
Authenticate as a GitHub App and get an installation access token.
"""

import json
import time
import subprocess
import sys
from pathlib import Path

# Using jwt library for token generation
try:
    import jwt
except ImportError:
    print("Installing PyJWT...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyJWT"])
    import jwt

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def create_jwt(app_id: int, private_key_path: str) -> str:
    """Create a JWT for GitHub App authentication."""
    with open(private_key_path, "r") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iat": now - 60,  # Issued at (60 seconds in the past for clock drift)
        "exp": now + (10 * 60),  # Expires in 10 minutes
        "iss": app_id,
    }

    return jwt.encode(payload, private_key, algorithm="RS256")


def get_installation_id(jwt_token: str, owner: str) -> int:
    """Get the installation ID for a specific owner."""
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.get(
        "https://api.github.com/app/installations",
        headers=headers,
    )
    response.raise_for_status()

    installations = response.json()
    for installation in installations:
        if installation["account"]["login"].lower() == owner.lower():
            return installation["id"]

    raise ValueError(f"No installation found for owner: {owner}")


def get_installation_token(jwt_token: str, installation_id: int) -> str:
    """Get an installation access token."""
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers,
    )
    response.raise_for_status()

    return response.json()["token"]


def main():
    app_id = 3225242
    private_key_path = "/Users/sune/Documents/References/claude-code-assistant-ppp.2026-03-30.private-key.pem"
    owner = "PleasePressPause"

    print(f"Creating JWT for App ID {app_id}...")
    jwt_token = create_jwt(app_id, private_key_path)

    print(f"Getting installation ID for {owner}...")
    installation_id = get_installation_id(jwt_token, owner)
    print(f"Installation ID: {installation_id}")

    print("Getting installation access token...")
    token = get_installation_token(jwt_token, installation_id)

    print("\n" + "=" * 50)
    print("SUCCESS! Use this token for git operations:")
    print("=" * 50)
    print(f"\nToken: {token[:20]}...{token[-10:]}")
    print(f"\nTo configure git, run:")
    print(f'  git remote set-url origin https://x-access-token:{token}@github.com/PleasePressPause/please-press-pause.git')
    print(f"\nOr export as environment variable:")
    print(f"  export GITHUB_TOKEN={token}")

    # Also output just the token for easy capture
    print(f"\n__TOKEN_START__{token}__TOKEN_END__")

    return token


if __name__ == "__main__":
    main()
