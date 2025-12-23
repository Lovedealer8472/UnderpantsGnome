"""
UI Module - Static UI v2 and print interception.
Handles all user interface rendering and output management.
"""

from .adapter import UIAdapter

# Compatibility: Re-export draw_panel from parent ui.py module
# This allows "from app.ui import draw_panel" to work even though
# draw_panel is defined in app/ui.py (not in this package)
import importlib.util
import sys
from pathlib import Path

# Get the parent directory (app/)
_parent_dir = Path(__file__).parent.parent
_ui_module_path = _parent_dir / "ui.py"

if _ui_module_path.exists():
    # Load ui.py as a module
    spec = importlib.util.spec_from_file_location("app.ui_module", _ui_module_path)
    if spec and spec.loader:
        ui_module = importlib.util.module_from_spec(spec)
        sys.modules["app.ui_module"] = ui_module
        spec.loader.exec_module(ui_module)
        # Re-export draw_panel
        if hasattr(ui_module, "draw_panel"):
            draw_panel = ui_module.draw_panel
        else:
            draw_panel = None
    else:
        draw_panel = None
else:
    draw_panel = None

__all__ = [
    "UIAdapter",
    "draw_panel",
]

