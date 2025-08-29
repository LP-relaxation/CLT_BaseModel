# For future developers

Technical notes
- After making changes, please make sure ALL tests in `tests/` folder pass, and add new tests for new code.
- Due to the highly complicated influenza model, there are a lot of input combinations and formats -- errors might arise due to incorrect inputs (e.g. dimensions, missing commas, etc...) -- if there is an error in running the model, the inputs should be checked first. Additionally, more work should be spent on writing input validators and error messages.

Tests to add
- Experiments
  - Make sure aggregating over subpopulation/age/risk is correct (e.g. in `get_state_var_df`).
  - Make sure all the ways to create different CSV files lead to consistent results!
- Accept-reject sampling
  - Reproducibility: running the algorithm twice (with the same RNG each time) should give the same result.
  - Make sure the sampling updates are applied correctly (e.g. to the correct subpopulation(s) and with the correct dimensions).

Features to add
- Would be nice to make the "checker" in `FluMetapopModel` `__init__` method more robust -- can check dimensions, check for nonnegativity, etc...