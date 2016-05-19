#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections

def data_layer(name):
        data_layer_str = '''name: "%s"
layer {
    name: "data"
    type: "Data"
    top: "data"
    top: "label"
    include {
        phase: TRAIN
    }
    transform_param {
        mirror: true
        crop_size: 224
        mean_value: 104
        mean_value: 117
        mean_value: 123
    }
    data_param {
        source: "../ilsvrc2012/ilsvrc2012_train"
        batch_size: 32
        backend: LMDB
    }
}
layer {
    name: "data"
    type: "Data"
    top: "data"
    top: "label"
    include {
        phase: TEST
    }
    transform_param {
        mirror: false
        crop_size: 224
        mean_value: 104
        mean_value: 117
        mean_value: 123
    }
    data_param {
        source: "../ilsvrc2012/ilsvrc2012_val"
        batch_size: 5
        backend: LMDB
    }
}

''' % name
        return data_layer_str

def conv_layer(kernel_size, num_output, stride, pad, name, bottom, top=None, filler="msra"):
    if top is None:
        top = name
    conv_layer_str = '''layer {{
    bottom: "{bottom}"
    top: "{top}"
    name: "{name}"
    type: "Convolution"
    convolution_param {{
        num_output: {num_output}
        kernel_size: {kernel_size}
        pad: {pad}
        stride: {stride}
        weight_filler {{
            type: "{filler}"
        }}
        bias_term: false
    }}
}}

'''.format(**locals())
    return conv_layer_str

def bn_layer(name, bottom, top):
    bn_layer_str = '''layer {{
    bottom: "{top}"
    top: "{top}"
    name: "bn{name}"
    type: "BatchNorm"
    param {{
        lr_mult: 0
    }}
    batch_norm_param {{
        use_global_stats: true
    }}
}}

layer {{
    bottom: "{top}"
    top: "{top}"
    name: "scale{name}"
    type: "Scale"
    scale_param {{
        bias_term: true
    }}
}}
'''.format(**locals())
    return bn_layer_str

def in_place_bn(name, activation):
    return bn_layer(name, activation, activation)

def pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "Pooling"
    pooling_param {
        pool: %s
        kernel_size: %d
        stride: %d
    }
}

'''%(bottom, top, layer_name, pool_type, kernel_size, stride)
    return pool_layer_str

def ave_pool(kernel_size, stride, layer_name, bottom):
    return pooling_layer(kernel_size, stride, 'AVE', layer_name, bottom, layer_name)

def fc_layer(layer_name, bottom, top, num_output=1000, filler="msra"):
    fc_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "InnerProduct"
    inner_product_param {
         num_output: %d
         weight_filler {
             type: "%s"
             std: 0.001
         }
         bias_filler {
             type: "constant"
             value: 0
         }
    }
}

'''%(bottom, top, layer_name, num_output, filler)
    return fc_layer_str

def eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
        eltwise_layer_str = '''layer {
    bottom: "%s"
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "Eltwise"
    eltwise_param {
        operation: %s
    }
}

'''%(bottom_1, bottom_2, top, layer_name, op_type)
        return eltwise_layer_str

def activation_layer(layer_name, bottom, top, act_type="ReLU"):
        act_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "%s"
}

'''%(bottom, top, layer_name, act_type)
        return act_layer_str

def in_place_relu(activation_name):
    return activation_layer(activation_name + '_relu', activation_name, activation_name, act_type='ReLU')

def softmax_loss(bottom):
        softmax_loss_str = '''layer {
    bottom: "%s"
    bottom: "label"
    name: "loss"
    type: "SoftmaxWithLoss"
    top: "loss"
}

layer {
    bottom: "%s"
    bottom: "label"
    top: "acc/top-1"
    name: "acc/top-1"
    type: "Accuracy"
    include {
        phase: TEST
    }
}

layer {
    bottom: "%s"
    bottom: "label"
    top: "acc/top-5"
    name: "acc/top-5"
    type: "Accuracy"
    include {
        phase: TEST
    }
    accuracy_param {
        top_k: 5
    }
}

'''%(bottom, bottom, bottom)
        return softmax_loss_str


def conv1_layers():
    layers = conv_layer(7, 64, 2, 0, 'conv1', 'data') \
        + in_place_bn('_conv1', 'conv1') \
        + in_place_relu('conv1') \
        + pooling_layer(3, 2, 'MAX', 'pool1', 'conv1', 'pool1')
    return layers

def normalized_conv_layers(conv_params, level, branch, prev_top, activation=True):
    """conv -> batch_norm -> ReLU"""

    name = '%s_branch%s' % (level, branch)
    activation_name = 'res' + name
    layers = conv_layer(*(conv_params + (activation_name, prev_top))) \
        + in_place_bn(name, activation_name)
    if activation:
        layers += in_place_relu(activation_name)
    return layers, activation_name

def bottleneck_layers(prev_top, level, num_output, bypass_activation=None, bypass_str='', bypass_stride=1):
    """1x1 -> 3x3 -> 1x1, with bypass and eltwise sum"""

    if bypass_activation is None:
        bypass_activation = prev_top
    all_layers = bypass_str
    layers, prev_top = normalized_conv_layers((1, num_output, bypass_stride, 0), level, '2a', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((3, num_output, 1, 1), level, '2b', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((1, num_output*4, 1, 0), level, '2c', prev_top, activation=False)
    all_layers += layers
    final_activation = 'res'+level
    all_layers += eltwise_layer(final_activation, bypass_activation, prev_top, final_activation) \
        + in_place_relu(final_activation)
    return all_layers, final_activation

def bottleneck_layer_set(
        prev_top,               # Previous activation name
        level,                  # Level number of this set, used for naming
        num_output,             # "num_output" param for most layers of this set
        num_bottlenecks,        # number of bottleneck sets
        bypass_params='default',    # Conv params of the bypass convolution 
        sublevel_naming='letters'): # Naming scheme of layer sets. MSRA sometimes uses letters sometimes numbers
    """A set of bottleneck layers, with the first one having an convolution bypass to accomodate size"""

    if bypass_params == 'default':
        bypass_params = (1, num_output*4, 2, 0)
    bypass_str, bypass_activation = normalized_conv_layers(bypass_params, '%da'%level, '1', prev_top, activation=False)
    network_str = ''
    if sublevel_naming == 'letters' and num_bottlenecks <= 26:
        sublevel_names = ascii_lowercase[:num_bottlenecks]
    else:
        sublevel_names = ['a'] + ['b' + str(i) for i in range(1, num_bottlenecks)]
    for index, sublevel in enumerate(sublevel_names):
        if index != 0:
            bypass_activation, bypass_str = None, ''
            layers, prev_top = bottleneck_layers(prev_top, '%d%s'%(level, sublevel), num_output, bypass_activation, bypass_str)
        else:
            layers, prev_top = bottleneck_layers(prev_top, '%d%s'%(level, sublevel), num_output, bypass_activation, bypass_str, bypass_params[2])
        network_str += layers
    return network_str, prev_top

def resnet(variant='50'): # Currently supports 50, 101, 152
    Level = collections.namedtuple('Level', ['level', 'num_bottlenecks', 'sublevel_naming'])
    Level.__new__.__defaults__ = ('letters',)

    network_str = data_layer('ResNet-' + variant)
    network_str += conv1_layers()
    prev_top = 'pool1'
    levels = {
        '50': (
            Level(2, 3),
            Level(3, 4),
            Level(4, 6),
            Level(5, 3),
        ),
        '101': (
            Level(2, 3),
            Level(3, 4, 'numbered'),
            Level(4, 23, 'numbered'),
            Level(5, 3),
        ),
        '152': (
            Level(2, 3),
            Level(3, 8, 'numbered'),
            Level(4, 36, 'numbered'),
            Level(5, 3),
        )
    }
    for level, num_bottlenecks, sublevel_naming in levels[variant]:
        if level == 2:
            bypass_params = (1, 256, 1, 0)
        else:
            bypass_params = 'default'
        layers, prev_top = bottleneck_layer_set(prev_top, level, 16*(2**level), num_bottlenecks, 
            bypass_params=bypass_params, sublevel_naming=sublevel_naming)
        network_str += layers
    network_str += ave_pool(7, 1, 'pool5', prev_top)
    network_str += fc_layer('fc1000', 'pool5', 'fc1000', num_output=1000, filler='gaussian')
    network_str += softmax_loss('fc1000')
    return network_str


def main():
    for net in ('50', '101', '152'):
        with open('ResNet_{}_train_val.prototxt'.format(net), 'w') as fp:
            fp.write(resnet(net))


if __name__ == '__main__':
    main()
