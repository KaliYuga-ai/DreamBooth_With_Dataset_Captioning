import argparse
import os
import tempfile

import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
sys.path.append('/content/EveryDream/scripts/BLIP/models')
from blip import blip_decoder
from utils.tokenizer import decode_batch_predictions

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='input', help='directory containing input images')
parser.add_argument('--temp_file', type=str, default='captions_temp.txt', help='temporary file to store generated captions')
parser.add_argument('--model_path', type=str, default='model_base_capfilt_large.pth', help='path for trained decoder model')
parser.add_argument('--image_size', type=int, default=384, help='size for center cropping images')
parser.add_argument('--num_beams', type=int, default=3, help='number of beams for beam search decoding')
parser.add_argument('--min_length', type=int, default=5, help='minimum length of the generated captions')
parser.add_argument('--max_length', type=int, default=20, help='maximum length of the generated captions')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for processing')

# Parse the arguments
args = parser.parse_args()

# Load the BLIP model
model = blip_decoder(pretrained=args.model_path, image_size=args.image_size, vit='base')
model.eval()
model.to(args.device)

# Define the image preprocessing steps
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
])

# Generate captions for all images in the directory
captions = []
for file_name in os.listdir(args.img_dir):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        file_path = os.path.join(args.img_dir, file_name)
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            caption = model.generate(image, sample=False, num_beams=args.num_beams, max_length=args.max_length, min_length=args.min_length)
        sentence = decode_batch_predictions(caption)[0]
        captions.append(f"{file_name}\t{sentence}")

# Write the captions to the temporary file
with open(args.temp_file, 'w') as f:
    f.write('\n'.join(captions))

print(f"Generated captions for {len(captions)} images.")
print(f"Captions stored in temporary file: {args.temp_file}")
