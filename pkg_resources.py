# pkg_resources.py
"""
Minimal compatibility shim for legacy code that does:
    from pkg_resources import parse_version

Newer packaging stacks may not expose pkg_resources as expected.
This shim provides parse_version via packaging.version.
"""
from packaging.version import parse as parse_version  # same name expected by caller
