from argparse import ArgumentParser
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers import CLIPModel
import torch
import onnx
import sys
from pathlib import Path


def build_argparser():
    parser = ArgumentParser()

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--model_dir', required=True,
                         help=f"Optional. Path to model weights")
    options.add_argument('-o', '--output', required=True,
                         help='Optional. Name of the output file(s) to save.')
    return parser


@torch.no_grad()
def convert_to_onnx(model, input_shapes, output_file):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    # output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_inputs = (torch.zeros(input_shapes[0], dtype=torch.float))
    model(torch.zeros(input_shapes[0], dtype=torch.float))
    torch.onnx.export(model, dummy_inputs, str(output_file), verbose=False, input_names=["input_image"], output_names=["output"])

    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)


def main():
    args = build_argparser().parse_args()

    model_text = CLIPVisionModelWithProjection.from_pretrained(args.model_dir)
    print(model_text)

    convert_to_onnx(model_text, input_shapes=[[1,3,224,224]], output_file=Path(args.output))


if __name__ == '__main__':
    sys.exit(main() or 0)
