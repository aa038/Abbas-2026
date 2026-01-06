This repository contains all the figures, scripts and data products associated with the paper Abbas et al. 2026

# Requirements and Installation

## Python version
This project was developed and tested with **Python 3.11**.  
Other Python 3.x versions may work, but 3.11 is recommended for reproducibility.  

## Local libraries
Three custom Python libraries are used throughout the analysis. They are included in this repository under the `src/` directory:

- `PlotStyle/` - shared plotting styles and figure utilities  
- `Orbituary/` - orbit fitting routine  
- `solve_orbit/` - routine to convert orbital elements into observable quantities (separations and position angles)

These must be installed in editable mode so updates are reflected automatically:

# Install local libraries  
```bash
pip install -e ./PlotStyle  
pip install -e ./Orbituary
pip install -e ./solve_orbit
``` 

## Python dependencies
The analysis also depends on common scientific Python packages, including:

- `numpy`  
- `pandas`  
- `matplotlib`  
- `pathlib`  

All dependencies are listed in `requirements.txt`.

# Install local libraries  
```bash
pip install -r requirements.txt  
```

## Clone the repository  
```bash
git clone https://github.com/aa038/Abbas-2026
cd Abbas-2026 
```

After this, all the scripts should run without import errors. 