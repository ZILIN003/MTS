import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
import warnings
from tqdm import tqdm
from PIL import Image
import pickle
warnings.filterwarnings("ignore")


# waterbirds celeba texture
root_path = 'data/waterbirds'
batch_size = 64

def load_image(path):
    return Image.open(path)

#######
# Use the same model_id as the model used to generate the data.
# For example, if the data was generated with Stable Diffusion v1.5,
# set model_id to "runwayml/stable-diffusion-v1-5".
#######

# model_id = "timbrooks/instruct-pix2pix"
# pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
model_id = 'runwayml/stable-diffusion-v1-5' #'runwayml/stable-diffusion-v1-5'  "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
vae = pipeline.vae.to("cuda")
image_processor = pipeline.image_processor


class_names = sorted([d for d in os.listdir(os.path.join(root_path, 'train'))])

def get_ebd(data_path, name=''):
    feature_dict = {}
    for cls_name in tqdm(class_names):
        feature_list = []
        images = []
        flag = 0
        cls_folder = os.path.join(os.path.join(root_path, data_path), cls_name)
        imgae_files = sorted(os.listdir(cls_folder))
        num_imgae_files = len(imgae_files)
        for fname in imgae_files:
            fpath = os.path.join(cls_folder, fname)
            images.append(Image.open(fpath).convert('RGB'))
            flag+=1
            if flag%batch_size==0 or flag==num_imgae_files:
                with torch.no_grad():
                    image_tensor = image_processor.preprocess(images, height=224, width=224).half().to("cuda")
                    image_latent = vae.encode(image_tensor).latent_dist.mode() #torch.Size([bz, 4, 28, 28])
                    pooling_feature = torch.mean(image_latent, dim=1).view(image_latent.shape[0], -1) #torch.Size([bz, 784])
                    feature_list.append(pooling_feature.cpu())
                    images = []

        class_feature = torch.cat(feature_list, dim=0)
        feature_dict[cls_name] = class_feature

    if data_path == 'train':
        pkl_file = 'real_ebd.pkl'
    else:
        pkl_file = f'syn_ebd{name}.pkl'

    with open(os.path.join(root_path, pkl_file), 'wb') as f:
        pickle.dump(feature_dict, f)

print(root_path)
get_ebd('train')
get_ebd('train_intervene')
# get_ebd('sketch', 'sketch')
# get_ebd('cartoon', 'cartoon')
# get_ebd('art_painting', 'art')


