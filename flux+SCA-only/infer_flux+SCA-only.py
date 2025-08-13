import argparse
from PIL import Image
import os
import torch
from tqdm import tqdm
from src.flux.xflux_pipeline import XFluxPipeline
import json

def create_argparser():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    return parser


def main(args):
    if args.image:
        image = Image.open(args.image)
    else:
        image = None

    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
    if args.use_ip:
        print('load ip-adapter:', args.ip_local_path, args.ip_repo_id, args.ip_name)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
    if args.use_lora:
        print('load lora:', args.lora_local_path, args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)
    if args.use_controlnet:
        print('load controlnet:', args.local_path, args.repo_id, args.name)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)
    
    # https://huggingface.co/datasets/SSS/ATR-bench/tree/main
    bench_path = '/path/to/bench.json'

    # Read JSON file and convert to dictionary
    with open(bench_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    seeds = [0]

    ip_scales = [0.6, 0.9]
     
    # ip_scales = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    total_iterations = len(data) * len(ip_scales) * len(seeds)

    with tqdm(total=total_iterations, desc="Generating images") as pbar:
        for index, entry in enumerate(data):
            word_content = entry['word']
            prompt = f'Text: {word_content}'
            img_prompt = entry['style_image']
            print(f"Index: {index}, Word: {prompt}, Style: {img_prompt}")

            image_prompt = Image.open(img_prompt)
            neg_image_prompt = Image.open(args.neg_img_prompt) if args.neg_img_prompt else None

            for ip_scale in ip_scales:
                for seed in seeds:
                    result = xflux_pipeline(
                        prompt=prompt,
                        controlnet_image=image,
                        width=args.width,
                        height=args.height,
                        guidance=args.guidance,
                        num_steps=args.num_steps,
                        seed=seed,
                        true_gs=args.true_gs,
                        control_weight=args.control_weight,
                        neg_prompt=args.neg_prompt,
                        timestep_to_start_cfg=args.timestep_to_start_cfg,
                        image_prompt=image_prompt,
                        neg_image_prompt=neg_image_prompt,
                        ip_scale=ip_scale,
                        neg_ip_scale=args.neg_ip_scale,
                    )
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)

                    result.save(os.path.join(args.save_path, f"img_index={index}_scale={ip_scale}_seed={seed}.png"))

                    prompt_filename = f"img_{index}.txt"
                    prompt_path = f"{args.save_path}/{prompt_filename}"
                    with open(prompt_path, 'w', encoding='utf-8') as file:
                        file.write(prompt)

                    pbar.update(1)


if __name__ == "__main__":
    
    output_img_path = f'/path/to/save/results'

    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)

    # Simulate command line arguments
    fake_argv = [
        "--ip_local_path", f"/path/to/checkpoint-only.safetensors", 
        "--device", "cuda:1",
        "--use_ip",
        "--width", "512",
        "--height", "512",
        "--timestep_to_start_cfg", "1",
        "--num_steps", "25",
        "--true_gs", "3.5",
        "--guidance", "4",
        "--save_path", f"{output_img_path}"
    ]
    
    # Simulate command line arguments
    import sys
    sys.argv = ['/path/to/FonTS/flux+SCA-only/infer_flux+SCA-only.py'] + fake_argv
    args = create_argparser().parse_args()
    main(args)
