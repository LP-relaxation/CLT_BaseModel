Notes
- "B&B 2025" refers to Bi & Bandekar et al's paper draft for influenza burden averted, using the immunoSEIRS model. 
- "CLT Lit Review" refers to the literature review that the whole group did to identify fixed values or reasonable ranges for various parameters.

Apologies -- the order that the parameter values are described in this README does not match the order given in the input files.

# `texas_common_params.json`

This `JSON` file has parameter values currently used in calibration attempts. A simple initial calibration attempt is to obtain a sensible `beta_baseline` (and potentially sensible population-level immunity parameter values and initial values) for one subpopulation.

Note that many of the fixed parameters will likely be common across city models, but some of them may be replaced by city-specific estimates from city-specific data sources. 

## Fixed parameters 
- `num_age_groups`, `num_risk_groups` -- we agreed that we have 5 age groups (0-4, 5-17, 18-49, 50-64, 65+).
- `total_contact_matrix` -- based on total contact matrix (contact rates, not probabilities) from [MOBS](https://github.com/mobs-lab/mixing-patterns) shared by Kaiming in March 2025 CLT meeting -- currently weighted by TEXAS population distribution among age groups -- city-level models may want to replace this with their city-specific population distributions, if available. See `README.md` in `derived_inputs_computation` folder for how this is computed. 
- `H_to_R_rate`, `H_to_D_rate` -- B+B 2025 and CLT Lit Review -- may eventually replace with city-specific hospitalization data
- `hosp_risk_reduce`, `inf_risk_reduce`, `death_risk_reduce` -- B+B 2025 
	- Assume `death_risk_reduce` is same as `hosp_risk_reduce`
- `IA_to_R_rate` -- CLT Lit Review (Remy)
- `IA_relative_inf` -- CLT Lit Review (Remy)
- `IP_relative_inf` -- CLT Lit Review (Remy)
- `E_to_I_rate` -- CLT Lit Review (Remy)
- `IP_to_IS_rate` -- CLT Lit Review (Remy)
- `IS_to_R_rate` -- CLT Lit Review (Remy)
- `E_to_IA_prop` -- CLT Lit Review (Remy)
- `IS_to_H_rate` -- CLT Lit Review (Sonny) -- but may eventually get age-specific rates from hospital data
- `H_to_R_rate` -- B&B 2025 -- also may eventually get from hospital data
- `H_to_D_rate` -- B&B 2025 and CLT Lit Review -- may eventually get from hospital data
- `IS_to_H_adjusted_prop` -— for the non-rate-adjusted proportion, used [2023-2024 CDC estimates](https://www.cdc.gov/flu-burden/php/data-vis/2023-2024.html#:~:text=The%20overall%20burden%20of%20influenza,and%2028%2C000%20flu%2Drelated%20deaths) and divided estimated hospitalizations by estimated infections for each age group —- this ends up being very similar to the table in the CLT Lit Review (Shraddha).
	- Then computed the rate-adjusted proportion using formula given in mathematical formulation.
    - Note: not sure if CDC methodology includes asymptomatic infections, and how much that affects our parameter value estimates — because `IS_to_H_adjusted_prop` is NOT the same as IHR because we are not considering asymptomatic people.
- `H_to_D_adjusted_prop` — for the non-rate-adjusted proportion, used [2023-2024 CDC estimates](https://www.cdc.gov/flu-burden/php/data-vis/2023-2024.html#:~:text=The%20overall%20burden%20of%20influenza,and%2028%2C000%20flu%2Drelated%20deaths) and used estimated deaths divided by estimated hospitalizations -— replaced Jose’s write-up for simplicity.
	- Again, computed the rate-adjusted proportion using formula given in mathematical formulation.

## Parameters that need to be fit or potentially re-assessed
- `beta_baseline` -- we are trying to fit this parameter. B&B 2025 list their calibrated value as `0.0493`.
- `R_to_S_rate` -- CLT Lit Review (Oluwasegun) -- will probably have to wiggle this for our new population-level immunity equations -- also, based on very preliminary calibration, it seems like this rate is too fast for sensible results
- `immune_saturation` -- based on what Anass said in April 2025 CLT meeting discussion, value was 100 from Covid-19 variant paper, but in our group meeting we agreed that immune parameters generally may need to be readjusted since we do have a new model and changed the way population-level immunity is updated, plus Covid-19 versus influenza is different -- another note: LP could not figure out how this was actually used in the immunoSEIRS burden averted paper code, so it seems like there is a discrepancy there.
- `hosp_immune_wane`, `inf_immune_wane` -- from B&B 2025 -- again, we will likely have to change this since we have a slightly different immunity update than previous immunoSEIRS versions. Note 1: LP does not really understand the units of this, and this is another reason we should probably revisit this. Note 2: based on CLT Lit Review (LP), there is a case to be made that these values are actually 0 within a given season, but based on March 2025 CLT meeting, Lauren wants immunity to go up and then down across the course of a flu season.

## Unused parameters
These parameters are not being included, at least in the first pass (simpler model for attempted calibration).

`humidity_impact` -- currently set to 0.
`hosp_immune_gain`, `inf_immune_gain` -- based on Lauren's decision after April 2025 CLT meeting discussion, we are not including these (setting them to 1 means they are not included).
`school_contact_matrix`, `work_contact_matrix` -- currently set to zero-matrices, so that we are not considering weekend or holiday seasonality (the force of infection includes the total contact matrix every day in the simulation).
`relative_suscept_by_age` -- set to all 1s, so not in effect currently. 

# `texas_compartments_epi_metrics_init_vals.json`

## Fixed parameters

- `S` -- population estimates from ACS 2023 1-year -- can also be taken from data source described in `README.md` in `derived_inputs_computation` folder.
- `R`, `D` -- starting off with zero-matrices.

## Parameters that need to be fit or potentially re-assessed

- `pop_immunity_inf`, `pop_immunity_hosp` -- B&B 2025 -- again, will likely have to change this because we have a new model.
- `E`, `IP`, `IS`, `IA`, `H` -- B&B 2025
	- Specific procedure: take total Texas population N, multiply by "initial proportion of the population in the infected compartment" (from B&B 2025), then multiply by "age-specific proportions of the initial infectious, exposed, and hospitalized compartments" (from B&B 2025) -- note that we have 3 infected compartments, so we just allocate the total infections equally among the infected compartments.
	- City-level models should adjust these initial values based on population counts in their city.
