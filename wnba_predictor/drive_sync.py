"""Optional Google Drive sync for the bet log and learned weights.

Mirrors what the Colab notebook did by hand (mount Drive, point the paths at
`MyDrive/wnba_prop_predictor`), but from a local script via OAuth instead of
Colab's built-in Drive mount. Entirely best-effort: every public function
catches its own errors and returns a status dict rather than raising, so the
app works fine with local-only files if Drive isn't set up (no
credentials.json) or is temporarily unreachable.

Setup (one-time, done by the user -- see README):
  1. Google Cloud Console -> new/existing project -> enable the Google Drive
     API -> OAuth client ID -> download as credentials.json into the repo
     root.
  2. Authorize once (scripts/get_drive_token.py, or Google's device-code
     flow for a terminal-free setup); the refresh token is cached to
     data/token.json (gitignored) so later runs don't prompt again.

Scope is the narrower `drive.file`, not the broader `drive` scope: it's
what Google's device-code flow (no local browser/redirect needed) actually
allows, and it's least-privilege besides -- the app can only see files and
folders it creates itself. Practical effect: the app creates its own
wnba_prop_predictor folder on first sync rather than reusing a
pre-existing folder of that name from another tool (e.g. an older Colab
notebook using its own Drive mount) -- drive.file has no way to discover a
folder it didn't create. If migrating history from such a folder, seed
data/bet_log.csv from it manually first; sync from then on is automatic.
"""

import io
import os
from typing import Optional

from . import config

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_FOLDER_NAME = "wnba_prop_predictor"

CREDENTIALS_PATH = os.environ.get("WNBA_DRIVE_CREDENTIALS", "credentials.json")
TOKEN_PATH = os.environ.get("WNBA_DRIVE_TOKEN", os.path.join(config.DATA_DIR, "token.json"))

SYNCED_FILENAMES = ["bet_log.csv", "learned_weights.json"]


def _status(enabled: bool, connected: bool, message: str) -> dict:
    return {"enabled": enabled, "connected": connected, "message": message}


def is_configured() -> bool:
    return os.path.exists(CREDENTIALS_PATH)


# Deploy platforms that inject secrets as read-only mounted files (e.g.
# Render's "Secret Files", mounted under /etc/secrets/) can't have their
# TOKEN_PATH rewritten in place after a refresh. Fall back to a writable
# cache alongside the bet log -- ephemeral disk is fine here since a
# refresh_token stays valid regardless of where the *access* token that
# results from using it gets cached.
_WRITABLE_TOKEN_CACHE = os.path.join(config.DATA_DIR, "_token_cache.json")


def get_service():
    """Returns an authenticated Drive API client, or raises if credentials.json
    is missing / the OAuth flow fails. Callers should catch and fail soft."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    # Prefer the writable cache if we've already refreshed once (it may be
    # newer than the original TOKEN_PATH, which could be a read-only mount).
    read_path = _WRITABLE_TOKEN_CACHE if os.path.exists(_WRITABLE_TOKEN_CACHE) else TOKEN_PATH

    creds = None
    if os.path.exists(read_path):
        creds = Credentials.from_authorized_user_file(read_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"No {CREDENTIALS_PATH} found. See README for how to create OAuth "
                    "credentials in Google Cloud Console, or skip Drive sync entirely."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Try the configured path first (works for normal local files);
        # if that's read-only (a mounted secret), fall back to the cache.
        for target in [TOKEN_PATH, _WRITABLE_TOKEN_CACHE]:
            try:
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                with open(target, "w") as f:
                    f.write(creds.to_json())
                break
            except OSError:
                continue

    return build("drive", "v3", credentials=creds)


def _find_folder(service, name: str) -> Optional[str]:
    query = (
        f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    res = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _create_folder(service, name: str) -> str:
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def _find_file(service, folder_id: str, filename: str) -> Optional[str]:
    query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    res = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _download(service, file_id: str, local_path: str) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(local_path, "wb") as f:
        f.write(buf.getvalue())


def _upload(service, folder_id: str, filename: str, local_path: str, existing_id: Optional[str]) -> str:
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(local_path, resumable=False)
    if existing_id:
        file = service.files().update(fileId=existing_id, media_body=media).execute()
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        file = service.files().create(body=metadata, media_body=media, fields="id").execute()
    return file["id"]


def sync_down(bet_log_path: str = config.BET_LOG_PATH,
              weights_path: str = config.LEARNED_WEIGHTS_PATH) -> dict:
    """Pulls bet_log.csv / learned_weights.json from Drive into local files,
    if Drive is configured and those files exist there. Meant to run once at
    app startup so a fresh machine picks up prior history."""
    if not is_configured():
        return _status(False, False, "credentials.json not found; Drive sync disabled")
    try:
        service = get_service()
        folder_id = _find_folder(service, DRIVE_FOLDER_NAME)
        if folder_id is None:
            return _status(True, True, f"no '{DRIVE_FOLDER_NAME}' folder in Drive yet; nothing to pull")

        pulled = []
        for filename, local_path in [("bet_log.csv", bet_log_path), ("learned_weights.json", weights_path)]:
            file_id = _find_file(service, folder_id, filename)
            if file_id:
                _download(service, file_id, local_path)
                pulled.append(filename)
        return _status(True, True, f"pulled from Drive: {', '.join(pulled) or 'nothing new'}")
    except Exception as e:
        return _status(True, False, f"Drive pull failed: {e}")


def sync_up(bet_log_path: str = config.BET_LOG_PATH,
            weights_path: str = config.LEARNED_WEIGHTS_PATH) -> dict:
    """Pushes local bet_log.csv / learned_weights.json up to Drive. Call this
    after any write (log_bet, record_outcome, recalibration) so Drive stays
    the durable copy across devices/sessions."""
    if not is_configured():
        return _status(False, False, "credentials.json not found; Drive sync disabled")
    try:
        service = get_service()
        folder_id = _find_folder(service, DRIVE_FOLDER_NAME)
        if folder_id is None:
            folder_id = _create_folder(service, DRIVE_FOLDER_NAME)

        pushed = []
        for filename, local_path in [("bet_log.csv", bet_log_path), ("learned_weights.json", weights_path)]:
            if not os.path.exists(local_path):
                continue
            existing_id = _find_file(service, folder_id, filename)
            _upload(service, folder_id, filename, local_path, existing_id)
            pushed.append(filename)
        return _status(True, True, f"pushed to Drive: {', '.join(pushed) or 'nothing to push'}")
    except Exception as e:
        return _status(True, False, f"Drive push failed: {e}")


def status() -> dict:
    if not is_configured():
        return _status(False, False, "not configured (no credentials.json)")
    connected = os.path.exists(TOKEN_PATH) or os.path.exists(_WRITABLE_TOKEN_CACHE)
    return _status(True, connected, "configured" + (", authorized" if connected else ", not yet authorized"))
