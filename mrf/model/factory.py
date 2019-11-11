import mrf.configuration.config as cfg

import mrf.model.midl as midl


MODEL_UNKNOWN_ERROR_MESSAGE = 'Unknown model "{}".'


def get_information(config: cfg.Configuration):
    if config.model == midl.MODEL_MIDL:
        return midl.MIDLModel, midl.MIDLModel.PADDING_SIZE
    else:
        raise ValueError(MODEL_UNKNOWN_ERROR_MESSAGE.format(config.model))


def get_model(config: cfg.Configuration):
    return get_information(config)[0]


def get_padding_size(config: cfg.Configuration):
    return get_information(config)[1]
