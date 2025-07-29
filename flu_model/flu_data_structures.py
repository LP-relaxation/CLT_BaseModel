import torch
from dataclasses import dataclass, fields, field


@dataclass
class FluMetapopStateTensors:

    S: torch.Tensor = None
    E: torch.Tensor = None
    IP: torch.Tensor = None
    IS: torch.Tensor = None
    IA: torch.Tensor = None
    H: torch.Tensor = None
    R: torch.Tensor = None
    D: torch.Tensor = None
    M: torch.Tensor = None
    Mv: torch.Tensor = None

    init_vals: dict = field(default_factory=dict)

    # Note: `init_vals: dict = {}` does NOT work --
    #   gives "mutable default" argument

    def save_current_vals_as_init_vals(self):

        for field in fields(self):
            if field.name == "init_vals":
                continue
            self.init_vals[field.name] = getattr(self, field.name).clone()

    def reset_to_init_vals(self):

        for name, val in self.init_vals.items():
            setattr(self, name, val.clone())


@dataclass
class FluMetapopParamsTensors:

    num_locations: int = None
    num_age_groups: int = None
    num_risk_groups: int = None

    beta_baseline: torch.Tensor = None
    humidity_impact: torch.Tensor = None

    R_to_S_rate: torch.Tensor = None
    E_to_I_rate: torch.Tensor = None
    IP_to_IS_rate: torch.Tensor = None
    IS_to_R_rate: torch.Tensor = None
    IA_to_R_rate: torch.Tensor = None
    IS_to_H_rate: torch.Tensor = None
    H_to_R_rate: torch.Tensor = None
    H_to_D_rate: torch.Tensor = None

    E_to_IA_prop: torch.Tensor = None
    H_to_D_adjusted_prop: torch.Tensor = None
    IS_to_H_adjusted_prop: torch.Tensor = None

    inf_induced_saturation: torch.Tensor = None
    inf_induced_immune_wane: torch.Tensor = None
    inf_induced_inf_risk_constant: torch.Tensor = None
    inf_induced_hosp_risk_constant: torch.Tensor = None
    inf_induced_death_risk_constant: torch.Tensor = None

    vax_induced_saturation: torch.Tensor = None
    vax_induced_immune_wane: torch.Tensor = None
    vax_induced_inf_risk_constant: torch.Tensor = None
    vax_induced_hosp_risk_constant: torch.Tensor = None
    vax_induced_death_risk_constant: torch.Tensor = None

    daily_vaccines: torch.Tensor = None

    total_contact_matrix: torch.Tensor = None
    school_contact_matrix: torch.Tensor = None
    work_contact_matrix: torch.Tensor = None

    IP_relative_inf: torch.Tensor = None
    IA_relative_inf: torch.Tensor = None

    relative_suscept: torch.Tensor = None
    prop_time_away: torch.Tensor = None

    travel_proportions_array: torch.Tensor = None

    def standardize_shapes(self) -> None:
        """
        If field is not a scalar or L x A x R, or is not a special variable
            listed below, then apply dimension expansion so that fields are
            L x A x R for tensor multiplication.

        Special variables that are exempted:
            - `total_contact_matrix`, `school_contact_matrix`,
                `work_contact_matrix` -- all of these must be dimension A x A
            - `travel_proportions_array`: this must be L x L

        Valid values for the `indices_dict` are: "age", "age_risk",
            "location", and "location_age" -- other combinations
            are not considered because they do not make sense --
            we assume that we only have risk IF we have age, for example
        """

        L = int(self.num_locations.item())
        A = int(self.num_age_groups.item())
        R = int(self.num_risk_groups.item())

        error_str = "Each SubpopParams field's size must be scalar or L x A x R " \
                    "(for location-age-risk groups)  -- please check files " \
                    "and inputs, then try again."

        for name, value in vars(self).items():

            # Ignore the field that corresponds to a dictionary
            if name == "init_vals":
                continue

            # Contact matrices should be A x A
            # This includes:
            #   "school_contact_matrix",
            #   "work_contact_matrix",
            #   "travel_proportions_array"
            elif "contact_matrix" in name:
                # Need nested if-statements because user may
                #   have already converted contact matrix to L x A x A
                if value.size() != torch.Size([L, A, A]):
                    if value.size() != torch.Size([A, A]):
                        raise Exception(str(name) + error_str)
                    setattr(self, name, value.view(1, A, A).expand(L, A, A))

            elif name == "travel_proportions_array":
                if value.size() != torch.Size([L, L]):
                    raise Exception(str(name) + error_str)

            # If scalar or already L x A x R, do not need to adjust
            #   dimensions
            elif value.size() == torch.Size([]):
                continue

            elif value.size() == torch.Size([L, A, R]):
                continue

            elif value.size() == torch.Size([L]):
                setattr(self, name, value.view(L, 1, 1).expand(L, A, R))


class FluPrecomputedTensors:
    """
    Stores precomputed quantities that are repeatedly
    used, for computational efficiency.
    """

    def __init__(self,
                 state: FluMetapopStateTensors,
                 params: FluMetapopParamsTensors) -> None:
        self.total_pop_LAR = torch.tensor(state.S +
                                          state.E +
                                          state.IP +
                                          state.IS +
                                          state.IA +
                                          state.H +
                                          state.R +
                                          state.D)

        self.L = int(params.num_locations.item())
        self.A = int(params.num_age_groups.item())
        self.R = int(params.num_risk_groups.item())

        self.total_pop_LA = torch.sum(self.total_pop_LAR, dim=2)

        # Remove the diagonal!
        self.nonlocal_travel_prop = params.travel_proportions_array.clone().fill_diagonal_(0.0)

        # We don't need einsum for residents traveling
        #   -- Dave and Remy helped me check this
        # \sum_{k \not = \ell} v^{\ell \rightarrow k}
        # Note we already have k \not = \ell because we set the diagonal of
        #   nonlocal_travel_prop to 0
        self.sum_residents_nonlocal_travel_prop = self.nonlocal_travel_prop.sum(dim=1)