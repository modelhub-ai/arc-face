import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
import mxnet as mx
import numpy as np
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # get context - cpu
        ctx = mx.cpu()
        image_size = (112,112)
        # Import ONNX model
        sym, arg_params, aux_params = import_model('model/model.onnx')
        # Define and binds parameters to the network
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self._model = model

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with mxnet
        input_blob = np.expand_dims(inputAsNpArr, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self._model.forward(db, is_train=False)
        embedding = self._model.get_outputs()[0].asnumpy()
        # postprocess results into output
        output = self._imageProcessor.computeOutput(embedding)
        return output.tolist()
