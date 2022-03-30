3k2f_22c.csv
-All but diffusivity values are predicted using PC-SAFT and DFT, as described in `incip_dft_string_fitting_code_guide.docx` in `Wang/dft_string_method/`.
-Diffusivity values are mostly copied from `3k2f_31c.csv` under the assumption that the counteracting factors of higher concentration of CO2 but higher viscosity at lower temperature roughly cancel each other out, leading to a roughly temperature-independent diffusivity (as seen when comparing 31 C and 60 C measurements with G-ADSA). The values in `3k2f_31c.csv` at the pressure nearest to the given pressure in `3k2f_22c.csv` was used (sensitivity to pressure is not so strong that this will lead to major errors, especially given uncertainty in measurements).
-Diffusivity at the highest pressure (~7000 kPa) was extrapolated with linear fit to intermediate pressure values (1000 kPa - 5500 kPa).

`clean_data.sh`: removes extraneous data and figures from image processing (`bubbletracking_koe` and `CvVidProc`)
To run, open Ubuntu, Bash, or Powershell, and run `./clean_data <directory with folders>` in this directory. Check out results by going to directory. Example directory = `ppg_co2/20211202_72bar`