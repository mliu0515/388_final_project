import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)

print(vis_processors.keys())

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print(model.generate({"image": image}))

print(model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3))

print(model.generate({"image": image, "prompt": "Question: what is this image about? Answer:"}))