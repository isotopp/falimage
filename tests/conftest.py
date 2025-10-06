import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure src/ is on sys.path for importing falimage package during tests
repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables from project .env for tests
load_dotenv(repo_root / ".env")
