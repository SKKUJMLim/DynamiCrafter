import sys
import subprocess
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_inference.py [256|512|1024]")
        sys.exit(1)

    version = sys.argv[1]
    seed = 123
    name = f"dynamicrafter_{version}_seed{seed}"

    ckpt = f"checkpoints/dynamicrafter_{version}_v1/model.ckpt"
    config = f"configs/inference_{version}_v1.0.yaml"
    prompt_dir = f"prompts/{version}/"
    res_dir = "results"

    # version별 설정
    if version == "256":
        H = 256
        FS = 3
    elif version == "512":
        H = 320
        FS = 24
    elif version == "1024":
        H = 576
        FS = 10
    else:
        print("Invalid input. Please enter 256, 512, or 1024.")
        sys.exit(1)

    # 공통 명령어
    cmd = [
        sys.executable, "scripts/evaluation/inference.py",
        "--seed", str(seed),
        "--ckpt_path", ckpt,
        "--config", config,
        "--savedir", f"{res_dir}/{name}",
        "--n_samples", "1",
        "--bs", "1",
        "--height", str(H),
        "--width", version,
        "--unconditional_guidance_scale", "7.5",
        "--ddim_steps", "50",
        "--ddim_eta", "1.0",
        "--prompt_dir", prompt_dir,
        "--text_input",
        "--video_length", "16",
        "--frame_stride", str(FS),
    ]

    # 512 / 1024 전용 옵션
    if version != "256":
        cmd.extend([
            "--timestep_spacing", "uniform_trailing",
            "--guidance_rescale", "0.7",
            "--perframe_ae"
        ])

    print("Running command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # python scripts/run_python.py 256
    main()
