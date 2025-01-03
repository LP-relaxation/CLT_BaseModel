from .base_components import Config, SimState, FixedParams
from .utils import np, json, Union, Type


def make_dataclass_from_json(dataclass_ref: Type[Union[Config, SimState, FixedParams]],
                             json_filepath: str) -> Union[Config, SimState, FixedParams]:
    """
    Create instance of class dataclass_ref,
    based on information in json_filepath.

    Args:
        dataclass_ref (Type[Union[Config, SimState, FixedParams]]):
            (class, not instance) from which to create instance.
        json_filepath (str):
            path to json file (path includes actual filename
            with suffix ".json") -- all json fields must
            match name and datatype of dataclass_ref instance
            attributes.

    Returns:
        Union[Config, SimState, FixedParams]:
            instance of dataclass_ref with attributes dynamically
            assigned by json_filepath file contents.
    """

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # convert lists to numpy arrays to support numpy operations
    #   since json does not have direct support for numpy
    for key, val in data.items():
        if type(val) is list:
            data[key] = np.asarray(val)

    return dataclass_ref(**data)