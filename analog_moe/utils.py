from typing import Any
import dill
import torch
from huggingface_hub import HfApi, hf_hub_download
from aihwkit.nn.conversion import convert_to_analog


def save_analog_model(model: Any, name: str, token: str = None):
    """
    Save an analog model to the hub.

    Example:
    ```python
        save_analog_model(model, name="ibm-aimc/analog-sigma-moe-small", token=token)
    ```

    Args:
        model (Any): Analog model to save.
        name (str): Name of the model for saving to the hub. Must be in the format "organization/model_name".
        token (str, optional): Token for writing to hub. Defaults to None.
    """
    rpu_config = next(model.analog_tiles()).rpu_config
    model.push_to_hub(name, token=token, safe_serialization=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dill.dumps(rpu_config),
        path_in_repo="rpu_config",
        repo_id=name,
        repo_type="model",
        token=token,
    )


def load_analog_model(
    name: str, fp_model_cls: Any, config_cls: Any, conversion_map: dict = None
):
    """
    Load an analog model from the hub.

    Args:
        name (str): Name for loading from the hub. Must be in the format "organization/model_name".
        fp_model_cls (Any): Class of the model in FP to load.
        config_cls (Any): Configuration class used for model.
        conversion_map (dict, optional): Dict mapping torch layers to analog layers. Defaults to None.

    Returns:
        Model: Loaded analog model.
    """
    path = hf_hub_download(repo_id=name, filename="pytorch_model.bin")
    rpu_path = hf_hub_download(repo_id=name, filename="rpu_config")
    with open(rpu_path, "rb") as f:
        rpu_config = dill.load(f)
    model_sd = torch.load(path)
    model = fp_model_cls(config=config_cls.from_pretrained(name))
    model = convert_to_analog(
        model,
        rpu_config=rpu_config,
        conversion_map=conversion_map,
    )
    out = model.load_state_dict(model_sd, strict=False)
    out.missing_keys
    for key in out.missing_keys:
        assert (
            "analog_tile_state" in key
        ), f"Error loading analog model. Missing key {key}. Are you sure the path to the FP model is correct?"
    return model