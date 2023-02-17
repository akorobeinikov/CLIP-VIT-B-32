# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


import logging as log
import onnxruntime as ort
import os
from openvino.runtime import Core, get_version, PartialShape, Dimension
import numpy as np
from typing import Any
import torch
import transformers

from .provider import ClassProvider
from .utils import get_model_path
from types import SimpleNamespace
from time import perf_counter


class BaseLauncher(ClassProvider):
    __provider_type__ = "launcher"
    def __init__(self, model_dir: str, model_name: str) -> None:
        """
        Load model using a model_name
        :param model_name
        """
        pass

    def process(self, input_arr: np.array) -> Any:
        """
        Run launcher with user's input
        :param input_arr
        """
        pass


class PyTorchLauncher(BaseLauncher):
    __provider__ = "pytorch"
    def __init__(self, models_dir: str, model_name: str) -> None:
        log.info('PyTorch Runtime')
        self.model_name = model_name
        if model_name == "text_model":
            self.model = transformers.CLIPTextModelWithProjection.from_pretrained(get_model_path(models_dir, self.__provider__))
        else:
            self.model = transformers.CLIPVisionModelWithProjection.from_pretrained(get_model_path(models_dir, self.__provider__))

    def process(self, input_arr: np.array) -> Any:
        outputs = self.model(torch.from_numpy(input_arr))
        if self.model_name == "text_model":
            return outputs.text_embeds
        else:
            return outputs.image_embeds


class ONNXLauncher(BaseLauncher):
    __provider__ = "onnx"
    def __init__(self, models_dir: str, model_name: str) -> None:
        log.info('ONNX Runtime')
        self.model_name = model_name
        self.session = ort.InferenceSession(os.path.join(models_dir, self.__provider__, model_name + ".onnx"))

    def process(self, input_arr: np.array) -> Any:
        if self.model_name == "text_model":
            outputs = self.session.run(["output"], {"input_text": input_arr})
        else:
            outputs = self.session.run(["output"], {"input_image": input_arr})
        finish_infer = perf_counter()
        return outputs[0]


class OpenVINOLaucnher(BaseLauncher):
    __provider__ = "openvino"
    def __init__(self, models_dir: str, model_name: str) -> None:
        log.info('OpenVINO Runtime')
        core = Core()
        self.model_name = model_name
        self.model = core.read_model(os.path.join(models_dir, self.__provider__, model_name, model_name + ".xml"))
        self.input_tensor = self.model.inputs[0].any_name
        # if not config.dynamic_shape and (self.model.inputs[0].partial_shape.is_dynamic or self.model.inputs[0].shape[1] != config.max_seq_len):
        if model_name == "text_model":
            self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(77)])})

        # if config.dynamic_shape:
        #     self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(0, config.max_seq_len)])})

        # load model to the device
        self.compiled_model = core.compile_model(self.model, "CPU")#{'CPU_THREADS_NUM': '48', 'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'})
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()

    def process(self, input_arr: np.array) -> Any:
        if self.model_name == "text_model":
            print(input_arr.shape)
            input_arr = np.pad(input_arr, ((0, 0), (0,77 - input_arr.shape[1])))
            print(input_arr.shape)
        inputs = {
            self.input_tensor: input_arr
        }
        # infer by OpenVINO runtime
        outputs = self.infer_request.infer(inputs)[self.output_tensor]
        return outputs


def create_launcher(laucnher_name: str, model_dir: str, model_name: str):
    return BaseLauncher.provide(laucnher_name, model_dir, model_name)
