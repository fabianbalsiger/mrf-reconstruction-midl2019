import tensorflow as tf

import mrf.configuration.config as cfg
import mrf.model.base as mdl_base


MODEL_MIDL = 'midl'


def layer(x, filters, kernel_size, dilation_rate, is_training_placeholder, dropout_p: float = 0, norm: str = 'none',
          activation='relu', layer_no: int = 1):
    x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                         padding='valid', activation=activation, name='layer{}_conv'.format(layer_no))

    layer_no += 1

    if dropout_p > 0:
        x = tf.layers.dropout(x, dropout_p, training=is_training_placeholder)

    if norm == 'bn':
        x = tf.layers.batch_normalization(x, training=is_training_placeholder)

    return x, layer_no


def dense_block_with_skip_connection(x, n_layers: int, growth_rate, kernel_size, dilation_rate, is_training_placeholder,
                                     dropout_p: float = 0, norm: str = 'none', activation='relu', layer_no: int = 1):
    for i in range(n_layers):
        out, layer_no = layer(x, growth_rate, kernel_size, dilation_rate, is_training_placeholder,
                              dropout_p, norm, activation, layer_no)
        x = tf.concat([x, out], axis=-1)

    return x, layer_no


class MIDLModel(mdl_base.MRFModel):

    PADDING_SIZE = (0, 5, 5)

    def __init__(self, session, sample: dict, config: cfg.Configuration, spatial_kernel_sizes=[3, 3, 3, 3, 3, 1]):
        self.spatial_kernel_sizes = spatial_kernel_sizes
        super().__init__(session, sample, config)

    def inference(self, x) -> object:
        x = tf.reshape(self.x_placeholder, (-1,
                                            self.x_placeholder.shape[1],
                                            self.x_placeholder.shape[2],
                                            self.x_placeholder.shape[3] * self.x_placeholder.shape[4]))

        spatial_kernel_idx = 0
        n_layers_in_dense_block = 4

        layer_no = 1
        x, layer_no = dense_block_with_skip_connection(x, n_layers_in_dense_block, 192, 1, 1,
                                                       self.is_training_placeholder, self.dropout_p, self.norm, 'relu',
                                                       layer_no)
        x, layer_no = layer(x, 192, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'relu', layer_no)
        spatial_kernel_idx += 1

        x, layer_no = dense_block_with_skip_connection(x, n_layers_in_dense_block, 160, 1, 1,
                                                       self.is_training_placeholder, self.dropout_p, self.norm, 'relu',
                                                       layer_no)
        x, layer_no = layer(x, 160, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'relu', layer_no)
        spatial_kernel_idx += 1

        x, layer_no = dense_block_with_skip_connection(x, n_layers_in_dense_block, 128, 1, 1,
                                                       self.is_training_placeholder, self.dropout_p, self.norm, 'relu',
                                                       layer_no)
        x, layer_no = layer(x, 128, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'relu', layer_no)
        spatial_kernel_idx += 1

        x, layer_no = dense_block_with_skip_connection(x, n_layers_in_dense_block, 96, 1, 1,
                                                       self.is_training_placeholder, self.dropout_p, self.norm, 'relu',
                                                       layer_no)
        x, layer_no = layer(x, 96, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'relu', layer_no)
        spatial_kernel_idx += 1

        x, layer_no = dense_block_with_skip_connection(x, n_layers_in_dense_block, 64, 1, 1,
                                                       self.is_training_placeholder, self.dropout_p, self.norm, 'relu',
                                                       layer_no)
        x, layer_no = layer(x, 64, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'relu', layer_no)
        spatial_kernel_idx += 1

        x, layer_no = layer(x, self.no_maps, self.spatial_kernel_sizes[spatial_kernel_idx], 1,
                            self.is_training_placeholder, self.dropout_p, self.norm, 'linear', layer_no)
        return tf.identity(x, name='network')
