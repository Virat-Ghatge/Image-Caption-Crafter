from transformers import BlipProcessor , BlipForConditionalGeneration
from PIL import Image
import gradio as gr

# loaoding the processor and model

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image , return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tolens="True")
    return caption

def caption_image(image):
    # takes a PIL image as input and passes it to the generate image function

    try :
        caption = generate_caption(image)
        return caption
    
    except Exception as e :
        return print(f"an error has occured {str(e)}")
    
iface = gr.Interface(
    inputs= gr.Image(type='pil'),
    fn = caption_image,
    outputs= "text",
    title= "Image captioning with gen-ai",
    description= "upload an image to generate caption for it"
)

iface.launch(server_name="127.0.0.1", server_port=5000 )
