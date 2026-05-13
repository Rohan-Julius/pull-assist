"""
Server registry for pull-assist CLI.

Uses a GitHub Gist as a central registry so:
  - Admin updates the GPU server URL with `pa admin set-gpu <URL>`
  - All CLI users auto-discover the current server — no IP sharing needed
  - When GPU is offline, users see "GPU not active, try later"

The Gist contains a JSON file with:
  {"server_url": "http://<IP>:9000/v1", "active": true, "message": ""}
"""

import json
import requests
from typing import Optional
from cli.config_manager import load_config, get_value

# ── Registry Gist ID ─────────────────────────────────────────────────────────
# HARDCODED — set once by admin with `pa admin init`, then committed to repo.
# All users who install the package automatically discover the GPU server.
# Users never need to know this value.
HARDCODED_GIST_ID = "fff9dc84577a30dd2505282fbe0a696a"  # <-- pa admin init fills this in automatically
REGISTRY_FILENAME = "pull-assist-registry.json"


def _get_gist_id() -> Optional[str]:
    """Get the registry Gist ID — hardcoded value takes priority."""
    if HARDCODED_GIST_ID:
        return HARDCODED_GIST_ID
    # Fall back to config (used by admin before committing the ID)
    return get_value("registry_gist_id") or None


def fetch_server_url() -> dict:
    """
    Fetch the current server URL from the registry.

    Returns:
        {
            "server_url": "http://...:9000/v1" or None,
            "active": True/False,
            "message": "..." (shown to user when inactive),
            "error": None or error string
        }
    """
    gist_id = _get_gist_id()

    if not gist_id:
        # No registry configured — fall back to local config
        local_server = get_value("server")
        if local_server and local_server != "http://localhost:8000/v1":
            return {
                "server_url": local_server,
                "active": True,
                "message": "",
                "error": None,
            }
        return {
            "server_url": None,
            "active": False,
            "message": "No server configured. Ask your admin for setup.",
            "error": "no_registry",
        }

    try:
        # Fetch the gist (public gists don't need auth)
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            timeout=10,
        )

        if resp.status_code == 404:
            return {
                "server_url": None,
                "active": False,
                "message": "Registry not found. Contact admin.",
                "error": "gist_not_found",
            }

        if resp.status_code != 200:
            return {
                "server_url": None,
                "active": False,
                "message": f"Registry error (HTTP {resp.status_code})",
                "error": f"http_{resp.status_code}",
            }

        gist_data = resp.json()
        files = gist_data.get("files", {})

        if REGISTRY_FILENAME not in files:
            return {
                "server_url": None,
                "active": False,
                "message": "Registry file missing. Contact admin.",
                "error": "file_missing",
            }

        content = json.loads(files[REGISTRY_FILENAME]["content"])

        return {
            "server_url": content.get("server_url"),
            "active": content.get("active", False),
            "message": content.get("message", ""),
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "server_url": None,
            "active": False,
            "message": "Could not reach GitHub (timeout). Check your internet.",
            "error": "timeout",
        }
    except requests.exceptions.ConnectionError:
        return {
            "server_url": None,
            "active": False,
            "message": "No internet connection.",
            "error": "no_internet",
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "server_url": None,
            "active": False,
            "message": f"Registry data corrupted: {e}",
            "error": "parse_error",
        }


def update_registry(server_url: str, active: bool = True,
                    message: str = "", github_token: str = None) -> dict:
    """
    Update the registry Gist with a new server URL.
    Only the admin should call this.

    Args:
        server_url: The GPU server URL (e.g. http://<IP>:9000/v1)
        active: Whether the GPU is currently active
        message: Message shown to users (e.g. "GPU available until 6pm")
        github_token: Admin's GitHub token (needs gist scope)

    Returns:
        {"ok": True/False, "error": str or None, "gist_id": str}
    """
    if not github_token:
        github_token = get_value("github_token")

    if not github_token:
        return {"ok": False, "error": "No GitHub token. Run: pa config set --token ghp_..."}

    gist_id = _get_gist_id()

    registry_content = json.dumps({
        "server_url": server_url,
        "active": active,
        "message": message,
    }, indent=2)

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        if gist_id:
            # Update existing gist
            resp = requests.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers=headers,
                json={
                    "files": {
                        REGISTRY_FILENAME: {"content": registry_content}
                    }
                },
                timeout=10,
            )
        else:
            # Create new gist
            resp = requests.post(
                "https://api.github.com/gists",
                headers=headers,
                json={
                    "description": "pull-assist server registry (auto-managed)",
                    "public": False,
                    "files": {
                        REGISTRY_FILENAME: {"content": registry_content}
                    }
                },
                timeout=10,
            )

        if resp.status_code in (200, 201):
            new_gist_id = resp.json()["id"]
            return {"ok": True, "error": None, "gist_id": new_gist_id}
        else:
            return {
                "ok": False,
                "error": f"GitHub API error {resp.status_code}: {resp.text[:200]}",
                "gist_id": gist_id,
            }

    except Exception as e:
        return {"ok": False, "error": str(e), "gist_id": gist_id}
