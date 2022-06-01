import tensorflow as tf


def extract(model, tflite_path, condition):
    tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
    tflite_interpreter.allocate_tensors()

    details = tflite_interpreter.get_tensor_details()

    for detail in details:
        if condition(detail['name']):
            save = tflite_interpreter.get_tensor(detail['index'])
