import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import gradio as gr
from PIL import Image

# ====== LLaVA 路径 ======
sys.path.insert(0, "/data/chengkaiwang/Project/touch_hallu/LLaVA-main")

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# =========================
# 全局：加载模型（只加载一次）
# =========================
MODEL_PATH = "/data/chengkaiwang/Project/touch_hallu/LLaVA-main/llava/train/checkpoints/stage2/llava-lora-merged"
MODEL_BASE = None
CONV_MODE = "llava_v1"

disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, MODEL_BASE, model_name
)
model = model.cuda().eval()


# =========================
# 推理函数（Gradio 调用）
# =========================
@torch.inference_mode()
def infer(
    image: Image.Image,
    prompt: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int
):
    if image is None:
        return "❌ 请先上传一张图片"

    # 1. 构造对话
    conv = conv_templates[CONV_MODE].copy()
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # 2. tokenizer
    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).cuda()

    # 3. image -> tensor
    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    # 4. generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p if temperature > 0 else None,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    # 5. decode
    output = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    return output


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Touch-LLaVA Inference") as demo:
    gr.Markdown(
        """
        # 🤖 Touch-LLaVA 推理 Demo
        上传触觉/接触图像，并输入描述或问题，模型将给出回答。
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="触觉图像 / 接触图像")
            prompt_input = gr.Textbox(
                label="Prompt",
                value="Please describe the tactile properties of the object.",
                lines=3
            )

            with gr.Accordion("⚙️ Generation Parameters", open=False):
                temperature = gr.Slider(0, 1.5, value=0.0, step=0.1, label="Temperature")
                top_p = gr.Slider(0, 1.0, value=0.9, step=0.05, label="Top-p")
                num_beams = gr.Slider(1, 5, value=1, step=1, label="Num Beams")
                max_new_tokens = gr.Slider(32, 256, value=100, step=16, label="Max Tokens")

            run_btn = gr.Button("🚀 Run Inference")

        with gr.Column():
            output_box = gr.Textbox(
                label="Model Output",
                lines=12,
                interactive=False
            )

    run_btn.click(
        fn=infer,
        inputs=[
            image_input,
            prompt_input,
            temperature,
            top_p,
            num_beams,
            max_new_tokens
        ],
        outputs=output_box
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
