import tensorflow as tf


def extract(model, tflite_path):
    tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
    tflite_interpreter.allocate_tensors()

    details = tflite_interpreter.get_tensor_details()

    # Get all the weights asociated with a name.
    weights_dict = {}
    errors = 0
    succeses = 0
    for detail in details:
        try:
            weights_dict[detail['name']] = tflite_interpreter.get_tensor(detail['index'])
            succeses += 1
        except ValueError:
            print("ValueError: ", detail['name'], end=' ')
            errors += 1
    print("\nFound {} weights and {} errors.".format(succeses, errors))

    ops = tflite_interpreter._get_ops_details()
    layers = iter(model.layers)
    for op in ops:
        # We only care about layers that have weights.
        if op['op_name'] not in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']:
            continue

        # Find the corresponding layer name for this op.
        layer = next(layers)
        while layer.name.find('conv') == -1:
            layer = next(layers)

        # Get the kernel for this layer.
        kernel_weight = weights_dict[details[op['inputs'][1]]['name'].replace("_dequantize", "")]
        if layer.name.find('depthwise_conv') != -1:
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
        elif layer.name.find('conv2d') != -1:
            kernel_weight = kernel_weight.transpose(1, 2, 3, 0)
        elif layer.name.find('conv') != -1:
            kernel_weight = kernel_weight.transpose(1, 0)

        # Get the bias for this layer.
        bias_weight = weights_dict[details[op['inputs'][2]]['name'].replace("_dequantize", "")]

        layer.set_weights([kernel_weight, bias_weight])
        print("Loaded weights for layer: ", layer.name, end='\r')

    print("\n")