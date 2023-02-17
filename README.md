# clip-ViT-B-32

## Hardware
For inference was used server with following parameters:
* CPU: 2 x Intel® Xeon® Gold 6248R
* RAM: 376 GB

## Models

| Model name        | Implementation   | Model card                                                                   |
|-------------------|------------------|------------------------------------------------------------------------------|
| clip-ViT-B-32     | PyTorch          | [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |


## Prerequisites

Before running inference, create python environment:
```
  python3 -m venv <env_name>
  source <env_name>/bin/activate
```

And install the necessary dependencies:
```
  python3 -m pip install -r requirements.txt
```


## Performance table

Results of performance experiments (1 Text / 1 Image):

| Model name        | Number of parameters   | Size of model | PyTorch time        | ONNXRuntime time | OpenVINO time |
|-------------------|------------------------|---------------|---------------------|------------------|---------------|
| clip-ViT-B-32     | -                      | 600 MB        | 20 ms / 40 ms       | 13 ms / 300 ms   | 30 ms / 50 ms |

## Benchmark results

| Model name                  | Async OpenVINO             | Sync OpenVINO                     | Sync ONNXRuntime          |
|-----------------------------|----------------------------|-----------------------------------|---------------------------|
| clip-ViT-B-32  text model   | FPS = 814, Latency = 58 ms | FPS = 123, Latency = 8 ms         | FPS = 90, Latency = 11 ms |
| clip-ViT-B-32  image model  | FPS = 500, Latency = 95 ms | FPS = 92, Latency = 11 ms         | FPS = 76, Latency = 13 ms |
