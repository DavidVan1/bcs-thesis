from pathlib import Path


def save_rpc_txt(rpc_dict: dict, output_file: Path):
    """Write RPC parameters to a text file.

    Args:
        rpc_dict: Dictionary containing RPC parameter names and values.
        output_file: Destination path for the RPC text file.

    Returns:
        None.
    """
    with open(output_file, "w") as f:
        for key, value in rpc_dict.items():
            if isinstance(value, list):
                f.write(f"{key}: {' '.join(map(str, value))}\n")
            else:
                f.write(f"{key}: {value}\n")

def compute_rpc(scene_dir: Path) -> dict:
    """Compute RPC parameters for a scene.

    Args:
        scene_dir: Input scene directory containing the data needed to derive RPCs.

    Returns:
        A dictionary with RPC metadata and coefficient arrays.
    """
    ### TODO: Implement the RPC computation here

    return {
        "LINE_OFF": 2048.0, "SAMP_OFF": 2048.0, "LAT_OFF": 0.0, "LONG_OFF": 0.0, "HEIGHT_OFF": 0.0,
        "LINE_SCALE": 2048.0, "SAMP_SCALE": 2048.0, "LAT_SCALE": 1.0, "LONG_SCALE": 1.0, "HEIGHT_SCALE": 500.0,
        "LINE_NUM_COEFF": [0.0] * 20, "LINE_DEN_COEFF": [1.0] + [0.0] * 19,
        "SAMP_NUM_COEFF": [0.0] * 20, "SAMP_DEN_COEFF": [1.0] + [0.0] * 19,
    }

def process_scene(scene_dir: Path, output_file: Path):
    """Entry point used by the evaluator to generate one scene's RPC file.

    Args:
        scene_dir: Input scene directory.
        output_file: Path where the RPC text file should be written.

    Returns:
        None.
    """
    rpc_results = compute_rpc(scene_dir)
    save_rpc_txt(rpc_results, output_file)