import gradio as gr
import os

# this is a demo of inference of IDEFICS-9B using 4bit-quantization which needs about 7GB of GPU memory
# which makes it possible to run even on Google Colab

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
#checkpoint = "HuggingFaceM4/tiny-random-idefics"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
processor = AutoProcessor.from_pretrained(checkpoint)

prompts = [
    "Instruction: provide an answer to the question. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
]


def response(input_image, input_question):
    input_prompt = [
    "Instruction: provide an answer to the question. Use the image to answer.\n",
    input_image,
    input_question
    ]

    inputs = processor(input_prompt, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=150)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_text[0]



with gr.Blocks() as demo:
    gr.Markdown("# IDEFICS")
    with gr.Row():
        image = gr.Pil()
        question = gr.Textbox(label="Question")
        submit_button = gr.Button("Submit")
    with gr.Column(scale=1):
        output = gr.Textbox(label="Output")
    
    submit_button_click = submit_button.click(fn=response, 
                                                       inputs=[image, question], 
                                                       outputs=[output])
    

if __name__ == "__main__":
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT'))) 

    print("Gradio app ready")