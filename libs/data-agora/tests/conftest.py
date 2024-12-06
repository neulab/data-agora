import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add the project root to the Python path
sys.path.insert(0, str(project_root))
