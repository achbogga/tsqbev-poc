# TSQBEV Container Images

This repo builds two OCI image targets:

- `ghcr.io/achbogga/tsqbev-poc:cpu`
- `ghcr.io/achbogga/tsqbev-poc:ray-gpu`
- `ghcr.io/achbogga/tsqbev-poc:cloud-demo`, an alias of `ray-gpu`

## Local CPU Validation

```bash
docker build -f docker/Dockerfile.cpu -t tsqbev-poc:cpu .
docker run --rm tsqbev-poc:cpu tsqbev smoke
```

## Local Ray GPU Build

```bash
docker build -f docker/Dockerfile.ray-gpu -t tsqbev-poc:cloud-demo .
```

GPU runtime validation requires a CUDA host:

```bash
docker run --gpus all --rm tsqbev-poc:cloud-demo tsqbev bench --device cuda
```

The `ray-gpu` image is based on `rayproject/ray:2.54.0-py311-gpu` and installs
TSQBEV with `uv` so Boba KubeRay RayJobs can reference
`ghcr.io/achbogga/tsqbev-poc:cloud-demo`.
