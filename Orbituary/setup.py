from setuptools import setup, find_packages

setup(
    name="orbituary",                    # Name of your package
    packages=find_packages(where="src"),  # Automatically find all packages in src/
    package_dir={"": "src"},             # Tell Python packages are in src directory
)