"""Local COMSOL paths and explicit checks for inputs omitted from this release.

COMSOL_BIN is the directory containing comsolbatch.exe. COMSOL_PREFS_FILE
may point to a user-supplied batch preference file. Neither is distributed.
The archived controllers retain their original input hashes and prerequisites.
"""
from pathlib import Path
import os


def require_file(path, description="archived input"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Historical MPH checkpoints, compiled "
            "COMSOL classes and private preferences are omitted from the public "
            "release; restore the required local input before using this controller.")
    return path


def comsol_bin():
    value = os.environ.get("COMSOL_BIN")
    if not value:
        raise RuntimeError(
            "COMSOL_BIN is not configured. Set it to the local COMSOL bin "
            "directory containing comsolbatch.exe; COMSOL is not bundled.")
    directory = Path(value).expanduser().resolve()
    require_file(directory / "comsolbatch.exe", "COMSOL batch executable")
    return directory


def comsol_batch():
    return comsol_bin() / "comsolbatch.exe"


def preferences_file(root):
    value = os.environ.get("COMSOL_PREFS_FILE")
    path = Path(value).expanduser().resolve() if value else Path(root) / "configs/comsol_batch_authorized.prefs"
    return require_file(path, "batch preferences (set COMSOL_PREFS_FILE)")


def require_prefs(directory):
    directory = Path(directory)
    require_file(directory / "comsol.prefs", "archived batch preferences")
    return directory
