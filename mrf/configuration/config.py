import pymia.config.configuration as cfg
import pymia.deeplearning.config as dlcfg

import mrf.data.data as data


class Configuration(dlcfg.DeepLearningConfiguration):
    """Represents a configuration."""

    VERSION = 1
    TYPE = 'MAIN'

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        """Initializes a new instance of the Configuration class."""
        super().__init__()
        self.indices_dir = ''
        self.split_file = ''

        self.model = ''  # string identifying the model
        self.experiment = ''  # string to describe experiment
        self.maps = [data.ID_MAP_T1H2O, data.ID_MAP_FF, data.ID_MAP_B1]  # the used maps
        self.patch_size = [1, 32, 32]

        # training configuration
        self.loss = 'mse'  # string identifying the loss function (huber, mse or mae)
        self.learning_rate = 0.01  # the learning rate
        self.dropout_p = 0.2
        self.norm = 'bn'  # none, bn

        # we use the mean absolute error as best model score
        self.best_model_score_is_positive = True
        self.best_model_score_name = 'mae'


def load(path: str, config_cls):
    """Loads a configuration file.

    Args:
        path (str): The path to the configuration file.
        config_cls (class): The configuration class (not an instance).

    Returns:
        (config_cls): The configuration.
    """

    return cfg.load(path, config_cls)
