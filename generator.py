#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections

def parse_args():
    parser = ArgumentParser(description=__doc__,
                                                    formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_val_file', help='Output train_val.prototxt file')
    parser.add_argument('--solver_file', help='Output solver.prototxt file',
                                            default='solver.prototxt')
    # parser.add_argument('--layer_number', nargs='*',
    #                                       help=('Layer number for each layer stage.'),
    #                                       default=[3, 8, 5, 3])
    # parser.add_argument('-t', '--type', type=int,
    #                                       help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
    #                                       default=1)

    args = parser.parse_args()
    return args

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
        batch_size: 8
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
    param {{
        lr_mult: 1
        decay_mult: 1
    }}
    param {{
        lr_mult: 2
        decay_mult: 0
    }}
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
    param {{
        lr_mult: 0
    }}
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
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
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
    name = '%s_branch%s' % (level, branch)
    activation_name = 'res' + name
    layers = conv_layer(*(conv_params + (activation_name, prev_top))) \
        + in_place_bn(name, activation_name)
    if activation:
        layers += in_place_relu(activation_name)
    return layers, activation_name

def bottleneck_layers(prev_top, level, num_output, bypass_activation=None, bypass_str=''):
    if bypass_activation is None:
        bypass_activation = prev_top
    all_layers = bypass_str
    layers, prev_top = normalized_conv_layers((1, num_output, 1, 0), level, '2a', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((3, num_output, 1, 1), level, '2b', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((1, num_output*4, 1, 0), level, '2c', prev_top, activation=False)
    all_layers += layers
    final_activation = 'res'+level
    all_layers += eltwise_layer(final_activation, bypass_activation, prev_top, final_activation) \
        + in_place_relu(final_activation)
    return all_layers, final_activation

def bottleneck_layer_set(prev_top, level, num_output, num_bottlenecks, bypass_params='default', sublevel_naming='letters'):
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


def solver(train_val_name):
        solver_str = '''net: "%s"
test_iter: 1000
test_interval: 6000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
stepvalue: 300000
stepvalue: 500000
gamma: 0.1
max_iter: 600000
momentum: 0.9
weight_decay: 0.0001
snapshot: 6000
snapshot_prefix: "resnet"
solver_mode: GPU
device_id: 0'''%(train_val_name)
        return solver_str

def main():
    args = parse_args()
    solver_str = solver(args.train_val_file)
    network_str = resnet('50')
    fp = open(args.solver_file, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(args.train_val_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
