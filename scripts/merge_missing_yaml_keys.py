#!/usr/bin/env python3

"""
Recursively add missing keys from a reference YAML into a current YAML.

- Does NOT overwrite existing keys
- Recurses into nested dictionaries
- Preserves existing values completely
- Adds only missing keys
- Safe for Neuralangelo / NeRF / ML configs

Usage:
    python merge_missing_yaml_keys.py reference.yaml current.yaml output.yaml

If output.yaml is omitted, overwrites current.yaml safely.
"""

import sys
import copy
from pathlib import Path
import yaml


def recursive_add_missing(ref, cur, path=""):
    """
    Recursively add missing keys from ref into cur.

    Parameters
    ----------
    ref : dict
        Reference YAML structure
    cur : dict
        Current YAML structure (modified in-place)
    path : str
        Debug path (for logging)

    Returns
    -------
    dict
        Updated current structure
    """

    # Only operate on dicts
    if not isinstance(ref, dict):
        return cur

    if not isinstance(cur, dict):
        return cur

    for key, ref_value in ref.items():

        if key not in cur:
            # Add missing key with deep copy
            cur[key] = copy.deepcopy(ref_value)

        else:
            # If both are dicts, recurse
            if isinstance(ref_value, dict) and isinstance(cur[key], dict):
                recursive_add_missing(ref_value, cur[key], path + "." + key)

    return cur


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False
        )


def main():

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    cur_path = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        out_path = Path(sys.argv[3])
    else:
        out_path = cur_path

    ref_yaml = load_yaml(ref_path)
    cur_yaml = load_yaml(cur_path)

    merged = recursive_add_missing(ref_yaml, cur_yaml)

    save_yaml(merged, out_path)

    print(f"✓ Missing keys added")
    print(f"Reference: {ref_path}")
    print(f"Current:   {cur_path}")
    print(f"Output:    {out_path}")


if __name__ == "__main__":
    main()