import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _strip_key_recursive(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            value = {k: v for k, v in value.items() if k != key}
        return {k: _strip_key_recursive(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_key_recursive(v, key) for v in value]
    return value


def patch_keras_file(path: Path, key_to_strip: str = "quantization_config") -> bool:
    if not path.exists() or path.suffix.lower() != ".keras":
        raise FileNotFoundError(path)

    # The .keras format is a zip file containing a config.json.
    with zipfile.ZipFile(path, "r") as zf:
        if "config.json" not in zf.namelist():
            return False
        original_config = json.loads(zf.read("config.json").decode("utf-8"))

    patched_config = _strip_key_recursive(original_config, key_to_strip)
    if patched_config == original_config:
        return False

    with zipfile.ZipFile(path, "r") as zf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_path, "w") as out:
                for info in zf.infolist():
                    if info.filename == "config.json":
                        data = json.dumps(patched_config, separators=(",", ":"), ensure_ascii=False).encode(
                            "utf-8"
                        )
                        out.writestr(info, data)
                    else:
                        out.writestr(info, zf.read(info.filename))

            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch .keras model files to improve cross-version loading.")
    parser.add_argument("path", type=Path, help="Path to a .keras file")
    args = parser.parse_args()

    changed = patch_keras_file(args.path)
    print(f"patched={changed} path={args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
