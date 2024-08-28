import subprocess
import sys
import venv
import os
import argparse

def main(env_dir, package_dir):
    # Create the virtual environment in root
    venv.create(env_dir, system_site_packages=False, with_pip=True)

    # Activate the virtual environment
    if sys.platform != "win32":
        activate_script = env_dir + '/bin/activate'
        activate_cmd = f'source {activate_script}'
    else:
        activate_script = env_dir + '/Scripts/activate.bat'
        activate_cmd = activate_script
    subprocess.run(activate_cmd, shell=True)

    if sys.platform != 'win32':
        venv_python = os.path.join(env_dir, 'bin', 'python') 
    else:
        venv_python = os.path.join(env_dir, 'Scripts', 'python.exe')

    subprocess.run([venv_python, '-m', 'pip', 'install', "-e", package_dir])

if __name__ =="__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Setup virtual environment and install package.')
    parser.add_argument('--env_dir', 
                        type=str, 
                        help='Path to the virtual environment')
    parser.add_argument('--package_dir', 
                        type=str, 
                        help='Path to the package to install')
    args = parser.parse_args()
    environment_dir = args.env_dir
    package_dir = args.package_dir

    # Default args
    if environment_dir is None:
        environment_dir = os.path.join(os.getcwd(), "env")

    if package_dir is None:
        package_dir = os.getcwd()

    # Run script
    main(environment_dir, package_dir)
