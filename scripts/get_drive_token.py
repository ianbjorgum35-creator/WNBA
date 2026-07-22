"""One-time helper: turns credentials.json into data/token.json.

Run this locally, once, after downloading credentials.json from Google
Cloud Console (OAuth client ID, type "Desktop app"). It opens a browser for
a one-time Google consent screen, then saves the resulting refresh token to
data/token.json. That file (plus credentials.json) is everything Drive sync
needs -- paste both into Render's Secret Files to enable Drive sync on a
deployed instance (see README, "Deploy to Render").

Usage:
    pip install google-auth-oauthlib google-api-python-client google-auth-httplib2
    python scripts/get_drive_token.py

Run it from the repo root (or anywhere with a `credentials.json` file next
to it) -- it does not need the rest of this project's dependencies.
"""

import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_PATH = os.environ.get("WNBA_DRIVE_CREDENTIALS", "credentials.json")
TOKEN_PATH = os.environ.get("WNBA_DRIVE_TOKEN", os.path.join("data", "token.json"))


def main():
    if not os.path.exists(CREDENTIALS_PATH):
        sys.exit(
            f"Could not find {CREDENTIALS_PATH}. Download it from Google Cloud Console "
            "(APIs & Services > Credentials > your OAuth client > Download JSON) and "
            "place it next to this script, or set WNBA_DRIVE_CREDENTIALS to its path."
        )

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    print("Opening a browser for Google sign-in... if it doesn't open automatically, "
          "copy the URL that gets printed below into your browser.")
    creds = flow.run_local_server(port=0)

    os.makedirs(os.path.dirname(TOKEN_PATH) or ".", exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print(f"\nSaved {TOKEN_PATH}. Open it and copy its contents into Render's "
          "token.json Secret File.")


if __name__ == "__main__":
    main()
