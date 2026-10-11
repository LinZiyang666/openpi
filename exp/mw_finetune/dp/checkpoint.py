"""Read inert checkpoint metadata without executing arbitrary pickle globals."""
import io
import json
from pathlib import Path
import pickle
import zipfile


class Nothing:
    def __new__(cls, *args, **kwargs):
        return {"__class__": "flax.nnx.filterlib.Nothing"}


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("flax.nnx.filterlib", "Nothing"):
            return Nothing
        raise pickle.UnpicklingError(f"Forbidden metadata global: {module}.{name}")


def read_metadata(checkpoint):
    root = Path(checkpoint)
    if (root / "metadata.json").exists():
        return json.loads((root / "metadata.json").read_text())
    with zipfile.ZipFile(root / "metadata.pt") as z:
        names = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("Expected one inert metadata pickle")
        return MetadataReader(io.BytesIO(z.read(names[0]))).load()


def read_stats(checkpoint):
    root = Path(checkpoint)
    paths = list(root.rglob("norm_stats.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one external normalization file")
    return json.loads(paths[0].read_text())["norm_stats"]
