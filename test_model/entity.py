class TFLiteModel():
    def __init__(self, model_file):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_file))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def output(self, img):
        if self.input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = self.input_details["quantization"]
            img = img / input_scale + input_zero_point

        img = np.expand_dims(img, axis=0).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details["index"], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details["index"])[0]