 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:13:33 2020

@author: ricard.deza.tripiana
"""
import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.onnx

from networks import utils
from networks.generator import GeneratorNet
from networks.discriminator import DiscriminatorNet
from networks.encoder import FeatureExtractor
from torch.autograd import Variable


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mse_loss = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss().to(device)

    # Inicialitzem el generador i el descriminadro
    generator = GeneratorNet().to(device)
    discriminator = DiscriminatorNet().to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Configurem el DataLoad
    transform = transforms.Compose([
        # transforms.Resize(args.image_size, Image.BICUBIC),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        # transforms.Lambda(lambda x: rgb2yuv(x)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.mul(255)),
        AddGaussianNoise(0., 1.)
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Configurem els optimitzadors
    optimizer_G = Adam(generator.parameters(), args.lr)
    optimizer_D = Adam(discriminator.parameters(), args.lr)
    
    Tensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    
    # Entrenament
    
    for e in range(args.epochs):
        for batch_id, (x, _) in enumerate(train_loader):
            
            # Configure input
            real_imgs = Variable(x.type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_imgs.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_imgs.size(0), *discriminator.output_shape))), requires_grad=False)

            # Entrenament del Generador
            generator.train()
            
            optimizer_G.zero_grad()

            # Generem un batch d'images
            # x = x.to(device)
            fake_imgs = generator(real_imgs)

            # Pèrdua que mesura la capacitat del generador per enganyar el discriminador
            adversarial_loss = mse_loss(discriminator(fake_imgs), valid)             
            
            # Perceptual loss
            gen_features = feature_extractor(fake_imgs)
            real_features = feature_extractor(real_imgs).detach()
            perceptual_loss = l1_loss(gen_features, real_features)
            
            alpha = 1
            # beta = 1
            
            loss_G = adversarial_loss + alpha*perceptual_loss #+ beta*diversity_loss
            
            loss_G.backward()
            optimizer_G.step()

            #  Entrenament del Discriminador
    
            optimizer_D.zero_grad()
    
            # Mesura de la habilitat del discriminador de classificar les imatges generades
            real_loss = mse_loss(discriminator(real_imgs), valid)
            fake_loss = mse_loss(discriminator(fake_imgs.detach()), fake)
            loss_D = 0.5 * (real_loss + fake_loss)


            loss_D.backward()
            optimizer_D.step()
    
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (e, args.epochs, batch_id, len(train_loader), loss_D.item(), loss_G.item())
            )
    
            batches_done = e * len(train_loader) + batch_id
            if batches_done % args.log_interval == 0:
                save_image(fake_imgs.data[:25], "/content/drive/My Drive/TFM/Generative Adversarial network/Artsy-gan/images/%d.png" % batches_done , nrow=5, normalize=True)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                generator.eval().cpu()
                ckpt_model_filename = "ckpt_generator_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(generator.state_dict(), ckpt_model_path)
                generator.to(device).train()
                
                discriminator.eval().cpu()
                ckpt_model_filename = "ckpt_discriminator_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(discriminator.state_dict(), ckpt_model_path)
                discriminator.to(device).train()

    # save model
    generator.eval().cpu()
    save_model_filename = "generator_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(generator.state_dict(), save_model_path)
    
    discriminator.eval().cpu()
    save_model_filename = "discriminator_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(generator.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = GeneratorNet().to(device)
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Artsy-GAN")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=200,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=1,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    # train_arg_parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
