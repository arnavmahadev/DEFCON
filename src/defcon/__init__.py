"""DEFCON — an unofficial reproduction of "Better Prevent than Tackle:
Valuing Defense in Soccer Based on Graph Neural Networks" (Kim et al.).

See ``tasks.md`` (git-ignored) for the phased implementation plan.
"""

__version__ = "0.1.0"

from defcon.config import Config, load_config  # noqa: E402,F401

__all__ = ["Config", "load_config", "__version__"]
