
# Derived Inputs Computations README 

This folder contains data from other public/academic sources that are used to derive some of our model's values. 

## Contact Matrix

(A) `United_States_subnational_Texas_M_overall_contact_matrix_85.csv` 
- From: https://github.com/mobs-lab/mixing-patterns/tree/main (access date 04/07/2025) -- see this github's `README` for file explanation
- File has same name as original file, found in data >> contact matrices folder

(B) `2023_ASRE_Texas_Population.csv`
- From: https://demographics.texas.gov/Estimates/2023/ -- Age, Sex, and Race/Ethnicity for State and Counties (access date 04/07/2025)
- File originally named `alldata.csv`

Code in `compute_texas_contact_matrix.py` is used to compute Texas contact matrix for the model -- specifically, the contact matrix given in (A) is aggregated into our model's age groups and then weighted by population given in (B).