from sentence_transformers import SentenceTransformer, util
from PIL import Image
from time import perf_counter

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

#Encode an image:
start_image = perf_counter()
img_emb = model.encode(Image.open('assets/dogs.jpg'))
finish_image = perf_counter()
print(f"Time to consider 1 image = {finish_image - start_image}")

#Encode text descriptions
start_text = perf_counter()
text_emb = model.encode(["a photo of a dog"])
finish_text = perf_counter()
print(f"Time to consider 1 text = {finish_text - start_text}")

#Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)