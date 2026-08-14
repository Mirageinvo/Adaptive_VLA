import argparse
import importlib
import math

from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


def get_cfg(cfg_path=None):
    parser = argparse.ArgumentParser()

    if cfg_path is None:
        parser.add_argument("--config", required=True, help="Path to config file")
        args, remaining_args = parser.parse_known_args()
        # laod yaml config file
        base_config = OmegaConf.load(args.config)
        OmegaConf.resolve(base_config)
        base_config.config = args.config
    else:
        args, remaining_args = parser.parse_known_args()
        base_config = OmegaConf.load(cfg_path)
        OmegaConf.resolve(base_config)
        base_config.config = cfg_path

    # e.g.: model.lr=0.01 train.batch_size=64
    cli_config = OmegaConf.from_cli(remaining_args)

    # merge config file and cli config, cli priority
    merged_config = OmegaConf.merge(base_config, cli_config)

    return merged_config


def load_cfg(config_path: str = None):
    # load yaml config file
    base_config = OmegaConf.load(config_path)
    OmegaConf.resolve(base_config)

    return base_config
