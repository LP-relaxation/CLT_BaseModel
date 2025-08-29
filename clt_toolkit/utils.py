from pathlib import Path
from dataclasses import replace, asdict, dataclass, is_dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def updated_dataclass(original: dataclass,
                      updates: dict) -> object:

    """
    Return a new dataclass based on `original`, with fields in `updates` replaced/added.
    """

    return replace(original, **updates)


def updated_dict(original: dict,
                 updates: dict) -> dict:
    """
    Return a new dictionary based on `original`, with keys in `updates` replaced/added.

    Parameters:
        original (dict):
            Original dictionary.
        updates (dict):
            Dictionary of updates to apply.

    Returns:
        (dict):
            New dictionary with updates applied.
    """

    return {**original, **updates}


def to_AR_array(x, A, R) -> np.ndarray:
    """
    Convert scalar, 1D (A,) or 2D (A,R) to a (A,R) array.

    Params:
        x (float | np.ndarray):
            Float or array to convert to (A, R) array.
        A (int):
            number of age groups.
        R (int):
            number of risk groups.

    Returns:
        (np.ndarray of shape (A, R))
    """

    arr = np.asarray(x)

    if arr.ndim == 0:  # scalar
        return np.full((A, R), arr)

    elif arr.ndim == 1:  # shape (A,)
        if arr.shape[0] != A:
            raise ValueError(f"Expected length {A}, got {arr.shape[0]}.")
        return np.tile(arr[:, None], (1, R))  # expand to (A,R)

    elif arr.ndim == 2:  # shape (A,R)
        if arr.shape != (A, R):
            raise ValueError(f"Expected shape ({A},{R}), got {arr.shape}.")
        return arr

    else:
        raise ValueError(f"Unsupported array shape {arr.shape}")


def daily_sum_over_timesteps(x: np.ndarray,
                             num_timesteps: int) -> np.ndarray:
    """
    For example, used for transition variable history, which is
    saved for every timestep, but we generally would like converted to daily totals.

    Params:
        x (np.ndarray of shape (N, A, R)):
            Array to aggregate -- N is the number of
            timesteps, A is the number of age groups,
            R is the number of risk groups.
        num_timesteps (int):
            Number of timesteps per day. Must divide
            N (length of `x`) evenly.

    Returns:
        (np.ndarray of shape (N/n, A, R):
            Array of daily totals, where each block of `num_timesteps`
            consecutive timesteps from `x` has been summed along the
            first dimension. The first axis now represents days instead
            of individual timesteps.
    """

    total_timesteps, A, R = x.shape

    if total_timesteps / num_timesteps != total_timesteps // num_timesteps:
        raise ValueError("x must be shape (N, A, R), where num_timesteps divides N.")

    num_days = int(total_timesteps/num_timesteps)

    # Pretty sweet hack ;)
    x = x.reshape(num_days, num_timesteps, A, R)

    # Sum along the number of timesteps
    return x.sum(axis=1)


def serialize_value(value):
    """
    Convert a value into a JSON-serializable format.

    Parameters:
    value (any):
        The value to serialize. Supported types:
        - `np.ndarray` is converted to `list`
        - Scalars and `None` remain unchanged
        - `dict`, `list`, or `tuple` gets recursively serialized
            (i.e. in case it's a nested object, etc...)
        - Any other type is converted to `str` as a fallback

    Returns:
        A version of the input that can be safely serialized to JSON.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [serialize_value(v) for v in value]
    else:
        return str(value)  # fallback for anything else


def serialize_dataclass(dc) -> dict:
    """
    Convert a dataclass or dict to a JSON-serializable dictionary.

    Parameters:
        dc (obj | dict):
            The object to serialize.
            - If a dataclass, it will be converted using `asdict()`.
            - All numpy arrays are converted to lists.
            - Scalars (int, float, str, bool) remain unchanged.
            - Other objects are converted to strings as a fallback.

    Returns:
        dict
            Dictionary representation of the object, fully JSON-serializable.
    """

    if is_dataclass(dc):
        dc = asdict(dc)
    elif not isinstance(dc, dict):
        raise TypeError("Object must be a dataclass or dict.")

    return {k: serialize_value(v) for k, v in dc.items()}