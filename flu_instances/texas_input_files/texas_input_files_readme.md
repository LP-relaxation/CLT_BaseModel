Notes
- "B&B 2025" refers to Bi & Bandekar et al's paper draft for influenza burden averted, using the immunoSEIRS model. 
- "CLT Lit Review" refers to the literature review that the whole group did to identify fixed values or reasonable ranges for various parameters.
- "Anass Analysis 2025" refers to new within-host model analysis conducted by Anass (using linear dM/dt differential equations)

Note: the order that the parameter values are described in this README may not necessarily match the order given in the input files.

# "school_work_calendar.csv"

***MUST BE UPDATED!*** This is taken from previous bespoke Austin model work (Dave Morton and Nazli Arslan), and is also from 2022-2023.

# "daily_vaccines_constant.csv"

***MUST BE UPDATED!*** This is a placeholder file -- notice that all "daily_vaccines" are 100. Must be updated with actual historical time series. 

# "absolute_humidity_austin_2023_2024.csv"

From Remy!

# "common_subpop_params.json"

This `JSON` file has parameter values currently used in calibration attempts. A simple initial calibration attempt is to obtain a sensible `beta_baseline` (and potentially sensible population-level immunity parameter values and initial values) for one subpopulation.

Note that many of the fixed parameters will likely be common across city models, but some of them may be replaced by city-specific estimates from city-specific data sources. 

## Fixed parameters 
- `num_age_groups`, `num_risk_groups` -- we agreed that we have 5 age groups (0-4, 5-17, 18-49, 50-64, 65+).
- `total_contact_matrix` -- based on total contact matrix (contact rates, not probabilities) from [MOBS](https://github.com/mobs-lab/mixing-patterns) shared by Kaiming in March 2025 CLT meeting -- currently weighted by TEXAS population distribution among age groups -- city-level models may want to replace this with their city-specific population distributions, if available. See `README.md` in `derived_inputs_computation` folder for how this is computed. 
- `HR_to_R_rate`, `HD_to_D_rate` -- B+B 2025 and CLT Lit Review -- may eventually replace with city-specific hospitalization data
- `hosp_risk_reduce`, `inf_risk_reduce`, `death_risk_reduce` -- B+B 2025 
    - Assume `death_risk_reduce` is same as `hosp_risk_reduce`
- `IA_to_R_rate` -- CLT Lit Review (Remy)
- `IA_relative_inf`, `IP_relative_inf` -- CLT Lit Review (Remy)
- `E_to_I_rate` -- CLT Lit Review (Remy)
- `IP_to_IS_rate` -- CLT Lit Review (Remy)
- `ISR_to_R_rate` -- CLT Lit Review (Remy)
- `E_to_IA_prop` -- CLT Lit Review (Remy)
- `ISH_to_H_rate` -- CLT Lit Review (Sonny) -- but may eventually get age-specific rates from hospital data
- `HR_to_R_rate` -- B&B 2025 -- also may eventually get from hospital data
- `HD_to_D_rate` -- B&B 2025 and CLT Lit Review -- may eventually get from hospital data
- `inf_induced_saturation`, `vax_induced_saturation` -- Anass Analysis 2025
- `inf_induced_inf_risk_reduce`, `vax_induced_inf_risk_reduce` -- Anass Analysis 2025
- `inf_induced_hosp_risk_reduce` -- Anass Analysis 08/09/2025, analysis in PDF posted in Slack: (the values of k/(k-1) equal) "6.81 for H1N1 and 9.09 for H3N2" -- according to the [CDC](https://www.cdc.gov/flu/whats-new/flu-summary-2023-2024.html) H1N1 was dominant in 2023-2024 so we'll use the first one -- solving for k, that means the risk reduction is 0.87 roughly.

## Fixed parameters that need to be double checked
- The values used in the json files (eg, common_subpop_params.json) for the following two parameters should be double checked. The values were updated using the CLT Lit Review when doing ghost compartment updates on 10/7/2025. It seems the calculation based on the sources below is happening somewhere else.
- `IP_to_ISH_prop` -— used [2023-2024 CDC estimates](https://www.cdc.gov/flu-burden/php/data-vis/2023-2024.html#:~:text=The%20overall%20burden%20of%20influenza,and%2028%2C000%20flu%2Drelated%20deaths) and divided estimated hospitalizations by estimated infections for each age group —- this ends up being very similar to the table in the CLT Lit Review (Shraddha).
    - Note: not sure if CDC methodology includes asymptomatic infections, and how much that affects our parameter value estimates — because `IP_to_ISH_prop` is NOT the same as IHR because we are not considering asymptomatic people.
- `ISH_to_HD_prop` — used [2023-2024 CDC estimates](https://www.cdc.gov/flu-burden/php/data-vis/2023-2024.html#:~:text=The%20overall%20burden%20of%20influenza,and%2028%2C000%20flu%2Drelated%20deaths) and used estimated deaths divided by estimated hospitalizations -— replaced Jose’s write-up for simplicity.

## Parameters that need to be changed or fit
- `mobility_modifier` -- need to be replaced by mobility data-driven estimates
- `beta_baseline` -- we are trying to fit this parameter. B&B 2025 list their calibrated value as `0.0493`
- `R_to_S_rate` -- CLT Lit Review (Oluwasegun) -- will probably have to wiggle this for our new population-level immunity equations -- also, based on very preliminary calibration, it seems like this rate is too fast for sensible results
- `inf_induced_immune_wane`, `vax_induced_immune_wane` -- CLT Lit Review (Linda, but had group discussion, and ended up using similar B&B 2025 values) -- specifically for infection-induced immunity waning, LP is suspicious because we only have literature on the half-life of antibodies, but that is not the same as half-life of protection against infection... may want to consider fitting these parameters.
- `vax_induced_hosp_risk_reduce` -- currently set to be same as `inf_induced_inf_risk_reduce` -- in the previous draft of the immunoSEIRS burden averted paper, there were no references for vaccination-induced protection against hospitalization -- Lauren says that Kaiming has new references -- someone should follow-up on this.

## Unused parameters
These parameters are not being included, at least in the first pass (simpler model for attempted calibration).
- `humidity_impact` -- currently set to 0.
- `school_contact_matrix`, `work_contact_matrix` -- currently set to zero-matrices, so that we are not considering weekend or holiday seasonality (the force of infection includes the total contact matrix every day in the simulation)
- `relative_suscept` -- set to all 1s, so not in effect currently.
- `inf_induced_death_risk_reduce`, `vax_induced_death_risk_reduce` -- 08/05/2025 meeting with Remy and Lauren -- Lauren said we are not including protection against death, so we set these to 0.

# `init_vals.json`

## Fixed parameters
- `S` -- Texas population estimates from ACS 2023 1-year -- can also be taken from data source described in `README.md` in `derived_inputs_computation` folder.
- `R`, `D` -- starting off with zero-matrices.

## Parameters that need to be changed or fit
- `E`, `IP`, `ISR`, `ISH`, `IA`, `HR`, `HD` -- B&B 2025
	- `E` for each age-risk group is simply set to 10, and the 3 infected compartments and `HR`, `HD` for each age-risk group are set to 1
	- City-level models should adjust these initial values based on population counts in their city.

# `mixing_params.json`

## Parameters that need to be changed or fit
- `num_locations` -- aka number of subpopulations -- this depends on the city
- `travel_proportions` -- need to be replaced by mobility data-driven estimates