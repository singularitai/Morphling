import warnings
import argparse
import os
from PIL import Image
import numpy as np
import torch
import random as r
import stylegan2
import matplotlib.pyplot as plt
from stylegan2 import utils

def generate_images(G, args):
    latent_size, label_size = G.latent_size, G.label_size
    device = torch.device('cpu')
    G.to(device)
    if args['truncation_psi'] != 1.0:
        G.set_truncation(truncation_psi=args['truncation_psi'])
    
    noise_reference = G.static_noise()

    def get_batch(seeds):
        latents = []
        labels = []

        noise_tensors = [[] for _ in noise_reference]
        for seed in seeds:
            rnd = np.random.RandomState(seed)
            latents.append(torch.from_numpy(rnd.randn(latent_size)))

            for i, ref in enumerate(noise_reference):
                noise_tensors[i].append(torch.from_numpy(rnd.randn(*ref.size()[1:])))
            if label_size:
                labels.append(torch.tensor([rnd.randint(0, label_size)]))
        latents = torch.stack(latents, dim=0).to(device=device, dtype=torch.float32)
        if labels:
            labels = torch.cat(labels, dim=0).to(device=device, dtype=torch.int64)
        else:
            labels = None

        noise_tensors = [
            torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)
            for noise in noise_tensors
        ]

        return latents, labels, noise_tensors

    for i in range(0, len(args['seed'])):
        latents, labels, noise_tensors = get_batch(args['seed'][i: i + 1])
        if noise_tensors is not None:
            G.static_noise(noise_tensors=noise_tensors)
        with torch.no_grad():
            generated = G(latents, labels=labels)
        images = utils.tensor_to_PIL(
            generated, pixel_min=-1.0, pixel_max=1.0)
        for seed, img in zip(args['seed'][i: i + 1], images):
            img.save(f"{args['output']}temp.png")

        
def main():
    
    args ={'output':'',
           'network':'models/stylegan_Gs.pth', 
           'seed': [int(r.uniform(0, (2**32 -1)))],
           'truncation_psi':r.uniform(0.7, 1.1)}

    G = stylegan2.models.load(args['network'])
    generate_images(G, args)
