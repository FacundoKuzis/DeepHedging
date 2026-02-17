import argparse
import os
import subprocess
import sys
import shutil


def get_venv_python(env_dir):
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def resolve_python311(user_python=None):
    if user_python:
        return user_python

    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                check=True,
            )
            candidate = proc.stdout.strip()
            if candidate and os.path.exists(candidate):
                return candidate
        except Exception:
            pass

    candidate = shutil.which("python3.11")
    if candidate:
        return candidate

    raise RuntimeError(
        "Python 3.11 was not found. Install Python 3.11 and run again "
        "(or provide --python path/to/python3.11)."
    )


def validate_python311(python_executable):
    proc = subprocess.run(
        [python_executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    )
    version = proc.stdout.strip()
    if version != "3.11":
        raise RuntimeError(
            f"Selected interpreter is Python {version}. This project requires Python 3.11."
        )


def create_venv(python_executable, env_dir):
    subprocess.run([python_executable, "-m", "venv", env_dir], check=True)


def ensure_existing_env_is_compatible(env_dir):
    if not os.path.isdir(env_dir):
        return

    existing_python = get_venv_python(env_dir)
    if not os.path.exists(existing_python):
        return

    proc = subprocess.run(
        [existing_python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    )
    existing_version = proc.stdout.strip()
    if existing_version != "3.11":
        raise RuntimeError(
            f"Existing virtual environment at '{env_dir}' is Python {existing_version}. "
            "Delete it and re-run to recreate it with Python 3.11."
        )


def main(env_dir, package_dir, python_executable=None):
    python_executable = resolve_python311(user_python=python_executable)
    validate_python311(python_executable)
    ensure_existing_env_is_compatible(env_dir)

    # Create the virtual environment with Python 3.11
    create_venv(python_executable, env_dir)

    venv_python = get_venv_python(env_dir)

    # Install the package in editable mode
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([venv_python, "-m", "pip", "install", "-e", package_dir], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup virtual environment and install package.")
    parser.add_argument("--env_dir", type=str, help="Path to the virtual environment")
    parser.add_argument("--package_dir", type=str, help="Path to the package to install")
    parser.add_argument("--python", type=str, help="Path to Python 3.11 executable")
    args = parser.parse_args()

    environment_dir = args.env_dir or os.path.join(os.getcwd(), "env")
    package_dir = args.package_dir or os.getcwd()

    main(environment_dir, package_dir, python_executable=args.python)
