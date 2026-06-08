try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

try:
    __version__ = version("fastyield")
except Exception:
    __version__ = "0+unknown"
