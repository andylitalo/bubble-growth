# bubble-growth

Collection of Jupyter notebooks and libraries of methods used to develop a
diffusion model of bubble growth. The model is based on the classic
Epstein-Plesset model of the growth of a bubble in a supersaturated liquid
(Epstein and Plesset, *J. Chem. Phys.* 1950) and has been adapted to incorporate
the effects of interfacial tension, changing pressure over time, and material-
specific parameters (*.e.g.*, the equation of state of the gas, gas solubility,
gas-liquid interfacial tension, and gas diffusivity in the liquid).

## Dependencies

These codes have been tested on Python 3.6. They require

``

## data

In the `data` folder, you will find the following files:
- `1k3f_30c.csv` gives material properties of high-pressure mixtures of
carbon dioxide and polypropylene glycol (PPG) 2700 g/mol measured with
gravimetry - axisymmetric drop-shape analysis (G-ADSA), a technique developed
in the Di Maio lab (U. Naples). The measurements were made at 30 C.
- `1k3f_22c.csv` gives the thermodynamic material properties from `1k3f_30c.csv`
predicted at 22 C (instead of 30 C) using the perturbed chain - statistical
associating fluid theory (PC-SAFT) and density functional theory (DFT) models,
whose parameters have been fit to the data in `1k3f_30c.csv` and `1k3f_60c.csv`
(the equivalent data measured at 60 C with G-ADSA). The diffusivity is also
included and comes from interpolating values from 1k3f_30c.csv since temperature
has a small effect on the diffusivity.
I also added a row at the beginning for $p = 0$ with the same values as at
$p = 100$ kPa to assist with interpolation for small values of presssure $p$.
The columns "co2 density [g/mL]" and before are processed using
`extract_dft_predictions.ipynb` jupyter notebook in the `g-adsa` folder
(different repository--please inquire if interested).

## Diffusion model

The original, complete diffusion model was created in
`20201124_diffn_matching.ipynb`. This model has been broken up into smaller
tasks for troubleshooting and optimization in `diffn_model_task#.ipynb`,
where `#` represents tasks 1, 1a, 2, and 3.
