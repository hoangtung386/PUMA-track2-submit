import subprocess
import sys


def test_public_api_imports_only_runtime_modules() -> None:
    forbidden = {"matplotlib", "pandas", "sklearn", "ultralytics"}
    command = f"import sys; import prometheus.api; assert not ({forbidden!r} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", command], check=True)
