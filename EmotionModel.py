from ngraph_onnx.onnx_importer.importer import import_onnx_file
import ngraph as ng
import numpy as np


class EmotionModel(object):
    def __init__(self):
        model = import_onnx_file("model/model.onnx")
        runtime = ng.runtime(backend_name="CPU")
        self.inference = runtime.computation(model)
        self.emotion_table = {
            "0": "neutral",
            "1": "happiness",
            "2": "surprise",
            "3": "sadness",
            "4": "anger",
            "5": "disgust",
            "6": "fear",
            "7": "contempt",
        }

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def _postprocess(self, x):
        prob = self._softmax(x)
        prob = np.squeeze(prob)
        classes = np.argsort(prob)[::-1]
        return {self.emotion_table[str(c)]: str(prob[c]) for c in classes}

    def predict(self, X, feature_names):
        return self._postprocess(self.inference(X))
