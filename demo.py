from api.utils import get_model_path
from api.launchers import create_launcher, BaseLauncher
from argparse import ArgumentParser
import logging as log
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from time import perf_counter
from transformers import CLIPTokenizerFast, CLIPImageProcessor
from typing import Union
import sys


log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

LAUNCHERS = [
    "pytorch",
    "onnx",
    "openvino"
]

def build_argparser():
    parser = ArgumentParser()

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--models_dir', required=True,
                         help=f"Optional. Path to model weights")
    options.add_argument('-l', '--launcher', required=False, choices=LAUNCHERS, default="pytorch",
                         help="Optional. Name of using backend for runtime. Available backends = {LAUNCHERS}. Default is 'PyTorch'")
    options.add_argument('-o', '--output', required=True,
                         help='Optional. Name of the output file(s) to save.')
    return parser


class CLIPModel:
    def __init__(self, models_dir, launcher: BaseLauncher) -> None:
        self.tokenizer = CLIPTokenizerFast.from_pretrained(get_model_path(models_dir, "pytorch"))
        self.image_processor = CLIPImageProcessor.from_pretrained(get_model_path(models_dir, "pytorch"))
        self.launcher = launcher

    def preprocess(self, input: Union[str, Image.Image]):
        if type(input) == str:
            preprocessed_input = self.tokenizer(input, return_tensors="np").input_ids
        else:
            preprocessed_input = self.image_processor(input, return_tensors="np").pixel_values

        return preprocessed_input

    def __call__(self, input: Union[str, Image.Image]):
        preprocessed_input = self.preprocess(input)
        return self.launcher.process(preprocessed_input)




def main():
    args = build_argparser().parse_args()

    image_launcher = create_launcher(args.launcher, args.models_dir, "image_model")
    text_launcher = create_launcher(args.launcher, args.models_dir, "text_model")

    # create models
    image_model = CLIPModel(args.models_dir, image_launcher)
    text_model = CLIPModel(args.models_dir, text_launcher)

    # run inference
    text_input = "a photo of a dog"
    image_input = Image.open('assets/dogs.jpg')

    start_text = perf_counter()
    text_emb = text_model(text_input)
    finish_text = perf_counter()
    print(f"Time to consider 1 text = {finish_text - start_text}")
    start_image = perf_counter()
    image_emb = image_model(image_input)
    finish_image = perf_counter()
    print(f"Time to consider 1 image = {finish_image - start_image}")

    #Compute cosine similarities
    cos_scores = util.cos_sim(image_emb, text_emb)
    print(cos_scores)


if __name__ == '__main__':
    sys.exit(main() or 0)
