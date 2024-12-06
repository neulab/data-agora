"""
Usage:

python -m alchemy.serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --gpu-ids 0 \
    --port 7000
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import huggingface_hub
from dotenv import load_dotenv

# Trouble Shooting: huggingface_hub.utils._errors.HfHubHTTPError: 416 Client Error: Requested Range Not Satisfiable for url
# Delete ~/.cache/huggingface/hub and try again
# ref: https://github.com/huggingface/huggingface_hub/issues/2197#issuecomment-2047170683


# ref: https://docs.ray.io/en/master/ray-core/objects/object-spilling.html#cluster-mode
def setup_ray_spilling(spill_dir: str = None):
    import json

    import ray

    if spill_dir is None:
        spill_dir = tempfile.mkdtemp(prefix="ray_spill_")
    else:
        spill_dir = Path(spill_dir).expanduser().resolve()
        spill_dir.mkdir(parents=True, exist_ok=True)

    ray.init(
        _temp_dir=str(spill_dir),
        _system_config={
            "object_spilling_config": json.dumps({"type": "filesystem", "params": {"directory_path": str(spill_dir)}}),
        },
    )
    return spill_dir


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run VLLM OpenAI-Compatible API server with specified arguments")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type for the model")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument(
        "--download-dir", type=str, default="~/.cache/huggingface/hub", help="Directory to download model"
    )
    parser.add_argument(
        "--gpu-ids", type=str, default="0", help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        "--ray-spill-dir",
        type=str,
        default=None,
        help="Directory for Ray object spilling (default: temporary directory)",
    )

    args = parser.parse_args()

    if args.ray_spill_dir is not None:
        spill_dir = setup_ray_spilling(args.ray_spill_dir)
        print(f"Ray object spilling directory: {spill_dir}")

    command = [
        f"CUDA_VISIBLE_DEVICES={args.gpu_ids}",
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--download-dir",
        args.download_dir,
    ]

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    subprocess.run(" ".join(command), shell=True)


if __name__ == "__main__":
    main()
