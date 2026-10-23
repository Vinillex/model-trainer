"""Put the clipboard's text into backend/.env as one setting, without ever printing it, then empty the clipboard.

For secrets shown once by a web page (an API key): press the page's Copy button, then run
    python -m scripts.env_from_clipboard MT_DATASET_BUCKET_SECRET_KEY
Only the setting's name and the value's length are printed. macOS only (pbpaste/pbcopy).
"""

import re
import subprocess
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def set_line(text: str, name: str, value: str) -> str:
    line = f"{name}={value}"
    pattern = re.compile(rf"(?m)^#?\s*{re.escape(name)}=.*$")
    if pattern.search(text):
        return pattern.sub(lambda _m: line, text, count=1)
    return text + ("" if text.endswith("\n") or not text else "\n") + line + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not re.fullmatch(r"MT_[A-Z0-9_]+", argv[0]):
        print("usage: python -m scripts.env_from_clipboard MT_SETTING_NAME")
        return 2
    name = argv[0]
    value = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True).stdout.strip()
    if not value or any(c.isspace() for c in value) or len(value) > 512:
        print(f"The clipboard doesn't hold a single value (length {len(value)}). Nothing was written.")
        return 1
    ENV.write_text(set_line(ENV.read_text() if ENV.exists() else "", name, value))
    subprocess.run(["pbcopy"], input="", text=True, check=True)  # don't leave the secret on the clipboard
    print(f"set {name} in {ENV.name} ({len(value)} characters); clipboard cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
