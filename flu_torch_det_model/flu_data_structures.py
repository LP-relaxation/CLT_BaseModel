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
    dt: float = None
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


class PrecomputedTensors:
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