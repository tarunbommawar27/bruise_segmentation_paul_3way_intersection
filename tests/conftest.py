"""Makes `import pipeline...` work when pytest is run from the project root,
without needing an installed package -- this repo has no setup.py/pyproject
(pipeline/ is imported via each script's own sys.path.insert instead), so
tests need the same project-root-on-sys.path treatment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
