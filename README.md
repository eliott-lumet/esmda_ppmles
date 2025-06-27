# ESMDA for parameter estimation in microscale pollutant dispersion 
This repository provides an example of Python implementation of the Ensemble Smoother with Multiple Data Assimilation (ESMDA) [1] algorithm. In this example, the algorithm is used to estimate meteorological forcing parameters (wind direction and friction velocity) using pollutant concentration measurements. The ESMDA can thereby improve the accuracy and robustness of an large-eddy simulation (LES) model, that reproduces the MUST field experiment [2]. To make the prediction step of the ESMDA afforable, the LES model is replaced by a POD--GPR surrogate model [3]. This reduces the prediction time by five orders of magnitude. This surrogate was previously trained using the PPMLES dataset [4] and is available on Zenodo (#TODO). 

### How to install?

Start by cloning the project

```bash
git clone git@github.com:eliott-lumet/esmda_ppmles.git
cd esmda_ppmles
```

Install a local python environment with all the dependencies

```bash
python -m venv .esmda_venv
source .esmda_venv/bin/activate
pip install .
```

You can now execute the `esmda.ipynb` jupyter notebook.

### References

[1] Emerick, A. A. and Reynolds, A. C. (2013). Ensemble smoother with multiple data assimilation. Computers & Geosciences, 55:3–15. DOI: [10.1016/j.cageo.2012.03.011](https://doi.org/10.1016/j.cageo.2012.03.011).

[2] Lumet, E. (2024)b. Assessing and reducing uncertainty in large-eddy simulation for microscale atmospheric dispersion. PhD thesis, Université Toulouse III - Paul Sabatier. URL: [https://theses.fr/2024TLSES003](https://theses.fr/2024TLSES003). Accessed: 2025-06-26.

[3] Lumet, E., Rochoux, M. C., Jaravel, T., and Lacroix, S. (2025). Uncertainty-Aware Surrogate Modeling for Urban Air Pollutant Dispersion Prediction. Building and Environment, page 112287. DOI: [10.1016/j.buildenv.2024.112287](https://doi.org/10.1016/j.buildenv.2024.112287).

[4] Lumet, E., Jaravel, T., and Rochoux, M. C. (2024)a. PPMLES – Perturbed-Parameter ensemble of MUST Large-Eddy Simulations. Dataset. Zenodo. DOI: [10.5281/zenodo.11394347](https://doi.org/10.5281/zenodo.11394347).
