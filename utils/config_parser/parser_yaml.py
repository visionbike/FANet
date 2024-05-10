import shutil
from abc import ABC
from pathlib import Path
import torch
from yacs.config import CfgNode as cn
from utils import get_class_weights
from utils.config_parser.parser_base import *

__all__ = ["ConfigParserYaml"]

VALID_TYPES = {tuple, list, str, int, float, bool, None, torch.Tensor}


def convert_to_dict(cfg_node, keys) -> dict:
    """
    Convert a configure node to dictionary.

    :param cfg_node: the input configuration node.
    :param keys: the list of keys.
    :return: a dictionary of CfgNode.
    """
    if not isinstance(cfg_node, cn):
        if type(cfg_node) not in VALID_TYPES:
            print(f"Key '{keys}' with value '{cfg_node}' is invalid.")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, keys + [k])
        return cfg_dict


class ConfigParserYaml(ConfigParserBase, ABC):
    """
    The custom configuration parser for YAML config files.
    """

    def __init__(self, description: str):
        """
        :paramdescription: the description for the parser.
        """
        super().__init__()
        self.parser.description = description
        # initial parser
        self.init_args()

    def _add_arguments(self):
        self.parser.add_argument("--cfg", default="./cfgs/config.yaml", help="the path of YAML config file.")

    def _print_args(self):
        super()._print_args()

    def init_args(self):
        self._add_arguments()

    def parse(self, cmd_args: list = None):
        """
        :param cmd_args: command arguments.
        :return: the output ConfigNode.
        """
        self.args = self.parser.parse_args(cmd_args)

        print("### Load the YAML config file...")
        with open(self.args.cfg, "r") as f:
            cfgs = cn.load_cfg(f)
            print(f"Successfully loading the config YAML file!")
        # initiate experiment configs
        exp_tags = []    # tags
        exp_name = ""    # exp name
        if "DataConfig" in cfgs.keys():
            # setup experiment name and config
            data_tags = cfgs.DataConfig.name.split("_")
            exp_tags += data_tags
            exp_name += cfgs.DataConfig.name
            # setup data name
            cfgs.DataConfig.name = data_tags[0]
            # setup num_classes
            if not cfgs.DataConfig.dataset_kwargs.use_rest_label:
                cfgs.DataConfig.num_classes -= 1
            # setup data validation mode
            cfgs.DataConfig.mode = cfgs.ExpConfig.split
        num_classes = cfgs.DataConfig.pop("num_classes")
        #
        if "NetworkConfig" in cfgs.keys():
            exp_name += "_" + cfgs.NetworkConfig.name + "_" + cfgs.NetworkConfig.attn_kwargs.name
            exp_tags += [cfgs.NetworkConfig.name]
            exp_tags += ["attn_" + cfgs.NetworkConfig.attn_kwargs.name if cfgs.NetworkConfig.attn_kwargs.name != "none" else cfgs.NetworkConfig.attn_kwargs.name]
            cfgs.NetworkConfig.num_classes = num_classes
        #
        if "LossConfig" in cfgs.keys():
            exp_tags += [cfgs.LossConfig.name]
            if cfgs.LossConfig.name == "focal":
                cfgs.LossConfig.alpha = get_class_weights(num_classes, cfgs.LossConfig.alpha)
            cfgs.LossConfig.num_classes = num_classes

        if "OptimConfig" in cfgs.keys():
            exp_tags += [cfgs.OptimConfig.name]

        if "LrSchedulerConfig" in cfgs.keys():
            exp_tags += [cfgs.LRSchedulerConfig.name]

        cfgs.MetricConfig = cn()
        cfgs.MetricConfig.num_classes = num_classes

        if "ExpConfig" in cfgs.keys():
            # name
            cfgs.ExpConfig.name = exp_name + "_" + cfgs.ExpConfig.split + "_exp" + str(cfgs.ExpConfig.experiment)
            cfgs.ExpConfig.tags = exp_tags + [cfgs.ExpConfig.split]
            # create "experiment" directory
            exp_path = Path("./experiments") / cfgs.ExpConfig.name
            exp_path.mkdir(parents=True, exist_ok=True)
            if not (exp_path / f"config.yaml").exists():
                shutil.copyfile(self.args.cfg, exp_path / f"config.yaml")
            # setup experiment configs
            cfgs.ExpConfig.exp_path = str(exp_path)

        # print configurations
        print(f"### Configurations:\n{cfgs}")
        return cfgs
