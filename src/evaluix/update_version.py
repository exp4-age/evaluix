# update_version.py
import re
from pathlib import Path

# Path to the __version__.py file
version_file = Path("src/evaluix/__version__.py")

# Read the version from __version__.py
with version_file.open() as f:
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

# Path to the pyproject.toml file
pyproject_file = Path("pyproject.toml")

# Read the pyproject.toml file
pyproject_content = pyproject_file.read_text()

# Replace the version in pyproject.toml
pyproject_content = re.sub(r'version = ".*"', f'version = "{version}"', pyproject_content)

# Write the updated content back to pyproject.toml
pyproject_file.write_text(pyproject_content)