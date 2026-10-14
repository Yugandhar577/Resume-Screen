"""Small, non-secret reproducibility record for explicit training runs."""
from importlib.metadata import version, PackageNotFoundError
import platform


def environment():
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU'}
    for package in ('torch','torch-geometric','transformers','sentence-transformers','numpy','pandas','pyarrow','scikit-learn'):
        try: result[package] = version(package)
        except PackageNotFoundError: pass
    return result
