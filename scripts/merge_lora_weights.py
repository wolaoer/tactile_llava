import argparse
import os,sys
sys.path.insert(0,"/data/chengkaiwang/Project/touch_hallu/LLaVA-main")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/chengkaiwang/Project/touch_hallu/LLaVA-main/llava/train/checkpoints/stage2/llava-lora-checkpoint-9951")
    parser.add_argument("--model-base", type=str, default="/data/chengkaiwang/Project/touch_hallu/LLaVA-1.1.3/weight/llava-7b")
    parser.add_argument("--save-model-path", type=str, default="/data/chengkaiwang/Project/touch_hallu/LLaVA-main/llava/train/checkpoints/stage2/llava-lora-merged-9951")

    args = parser.parse_args()

    merge_lora(args)
