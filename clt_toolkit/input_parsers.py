import numpy as np
import json
from typing import Type
from typing import Protocol


class DataClassProtocol(Protocol):
    __dataclass_fields__: dict


def convert_dict_vals_lists_to_arrays(d: dict) -> dict:
    """
    Converts dictionary of lists to dictionary of arrays
    to support `numpy` operations.
    """

    for key, val in d.items():
        if type(val) is list:
            d[key] = np.asarray(val)

    return d


def load_json_new_dict(json_filepath: str) -> dict:
    """
    Loads specified `JSON` file into new dictionary.
    Lists are automatically converted to numpy arrays for
    computational compatibility, since `JSON` does not natively
    support `np.ndarray`.

    Args:
        json_filepath (str):
            Full `JSON` filepath.

    Returns:
        (dict):
            Dictionary loaded with `JSON` information.
    """

    # Note: the "with open" is important for file handling
    #   and avoiding resource leaks -- otherwise,
    #   we have to manually close the file, which is a bit
    #   more cumbersome
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # json does not support numpy, so we must convert
    #   lists to numpy arrays
    return convert_dict_vals_lists_to_arrays(data)


def load_json_augment_dict(json_filepath: str,
                           d: dict) -> dict:
    """
    Augments pre-existing dictionary with information
    from `JSON` file -- if keys already exist, the previous values
    are overriden, otherwise the new key-value pairs are added.
    Lists are automatically converted to numpy arrays for
    computational compatibility, since `JSON` does not natively
    support `np.ndarray`.

    Args:
        json_filepath (str):
            Full `JSON` filepath.
        d (dict):
            Dictionary to be augmented with new `JSON` values.

    Returns:
        (dict):
            Dictionary loaded with `JSON` information.
    """

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    data = convert_dict_vals_lists_to_arrays(data)

    for key, val in data.items():
        d[key] = val

    return d


def make_dataclass_from_dict(dataclass_ref: Type[DataClassProtocol],
                             d: dict) -> DataClassProtocol:
    """
    Create instance of class dataclass_ref,
    based on information in dictionary.

    Args:
        dataclass_ref (Type[DataClassProtocol]):
            (class, not instance) from which to create instance --
            must have dataclass decorator.
        d (dict):
            all keys and values respectively must match name and datatype
            of dataclass_ref instance attributes.

    Returns:
        DataClassProtocol:
            instance of dataclass_ref with attributes dynamically
            assigned by json_filepath file contents.
    """

    d = convert_dict_vals_lists_to_arrays(d)

    return dataclass_ref(**d)


def make_dataclass_from_json(json_filepath: str,
                             dataclass_ref: Type[DataClassProtocol]) -> DataClassProtocol:
    """
    Create instance of class dataclass_ref,
    based on information in json_filepath.

    Args:
        json_filepath (str):
            path to json file (path includes actual filename
            with suffix ".json") -- all json fields must
            match name and datatype of dataclass_ref instance
            attributes.
        dataclass_ref (Type[DataClassProtocol]):
            (class, not instance) from which to create instance --
            must have dataclass decorator.

    Returns:
        DataClassProtocol:
            instance of dataclass_ref with attributes dynamically
            assigned by json_filepath file contents.
    """

    d = load_json_new_dict(json_filepath)

    return make_dataclass_from_dict(dataclass_ref, d)

