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
from matplotlib import pyplot
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.onnx

from networks import utils
from networks.generator_wchr import GeneratorNet
from networks.discriminator import DiscriminatorNet
from networks.encoder import FeatureExtractor

# Funció que chequeja la existencia de les carpetes on es desen el models
# tan de checkpoint com final. En el cas de no existir, les crea
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir + args.dataset_name):
            os.makedirs(args.save_model_dir + args.dataset_name)
    except OSError as e:
        print(e)
        sys.exit(1)

# Classe que afegeix soroll gaussià a una imatge. Per defecte desviació 1 i mitjana 0
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Funció per crear un gràfic amb la evolució de les pèrdues
def plot_history(d1_hist, g1_hist, g2_hist, g3_hist, g4_hist):
    # plot loss
    pyplot.subplot(5, 1, 1)
    pyplot.plot(d1_hist, label='d-loss')
    pyplot.legend()
    pyplot.subplot(5, 1, 2)
    pyplot.plot(g1_hist, label='g-adversarial')
    pyplot.legend()
    pyplot.subplot(5, 1, 3)
    pyplot.plot(g2_hist, label='g-perceptual')
    pyplot.legend()
    pyplot.subplot(5, 1, 4)
    pyplot.plot(g3_hist, label='g-diversity')
    pyplot.legend()
    pyplot.subplot(5, 1, 5)
    pyplot.plot(g3_hist, label='g-loss')
    pyplot.legend()
    
    # save plot to file
    # pyplot.savefig('results_opt/plot_line_plot_loss_wchr.png')
    pyplot.savefig('/content/drive/My Drive/TFM/Generative Adversarial network/Artsy-gan/results_opt/plot_line_plot_loss_wchr.png')
    pyplot.close()

# Funció d'entrenament del model de transferència d'estil
def train(args):
    # Inicialitzem la unitat de processament seleccionada
    device = torch.device("cuda" if args.cuda else "cpu")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Definim les pèrdues definides en l'entrenament
    mse_loss = torch.nn.MSELoss().to(device)
    l1_loss = torch.nn.L1Loss().to(device)

    # Inicialitzem el generador i el descriminador
    generator = GeneratorNet().to(device)
    discriminator = DiscriminatorNet().to(device)
    feature_extractor = FeatureExtractor().to(device)
    
    if args.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("models/%s/generator_epoch_%d_wchr.pth" % (args.dataset_name, args.epoch), map_location=device))
        discriminator.load_state_dict(torch.load("models/%s/discriminator_epoch_%d_wchr.pth" % (args.dataset_name, args.epoch), map_location=device)['model_state_dict'])
        # generator.load_state_dict(torch.load("/content/drive/My Drive/TFM/Generative Adversarial network/Artsy-gan/models/%s/generator_epoch_%d_wchr.pth" % (args.dataset_name, args.epoch)))
        # discriminator.load_state_dict(torch.load("/content/drive/My Drive/TFM/Generative Adversarial network/Artsy-gan/models/%s/discriminator_epoch_%d_wchr.pth" % (args.dataset_name, args.epoch)))

    #  Es posa l'extractor de característiques en mode d'inferència
    feature_extractor.eval()

    # Configurem les següents transformacions:
        # Redimensionem les imatges amb el paràmetre d'entrada image_size
        # Retallem el centre de la imatges amb la mida anterior
        # Convertim la imatges d'entrada en un Tensor
        # Normalitzem el Tensor
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Lambda(lambda x: x.mul(255)),
    ])
    
    # Carreguem les imatges en la carpeta dataset i apliquem les transformacions
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    # Creem un iterable soble el conjunt d'imatges amb un batch definit per paràmetre 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Configurem els optimitzadors
    optimizer_G = Adam(generator.parameters(), args.lr, betas=(args.b1, args.b2))
    optimizer_D = Adam(discriminator.parameters(), args.lr, betas=(args.b1, args.b2))
    
    # Programadors d’actualització de les taxes d’aprenentatge
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=utils.LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=utils.LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
    )
    
    Tensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    
    # Preparem les llistes per emmagatzemar les estadístiques de cada iteració
    d1_hist, g1_hist, g2_hist, g3_hist, g4_hist = list(), list(), list(), list(), list()
   
    # Entrenament
    # Per cada epoch
    for epoch in range(args.epoch, args.n_epochs):
        # Per cada batch del Data loader
        for batch_id, (x, _) in enumerate(train_loader):
            # Preparem el batch d'entrada amb un "embolcall" en Variables
            real_imgs = Variable(x.type(Tensor))
            
            # Creem les veritat fonamentals adversarials
            valid = Variable(Tensor(np.ones((real_imgs.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_imgs.size(0), *discriminator.output_shape))), requires_grad=False)
            
            #############################
            #   Entrenem el Generador   #
            #############################
            
            generator.train()

            # Inicialitzem l'optimitzador del generador
            optimizer_G.zero_grad()
            
            # Passem el batch d'entrada pel generador
            fake_imgs = generator(real_imgs)
            
            # CALCULEM LA PÈRDUA ADVERSARIAL
            # Pèrdua que mesura la capacitat del generador per enganyar el discriminador
            adversarial_loss = mse_loss(discriminator(fake_imgs), valid)             
            
            # CALCULEM LA PÈRDUA PERCEPTUAL
            # Generem el batch per calcular la pèrdua
            fake_imgs_perceptual = generator(real_imgs)
            # Extraiem les característiques del batch generat pel generador
            gen_features = feature_extractor(fake_imgs_perceptual)
            # Preparem el batch d'entrada amb un "embolcall" en Variables
            real_imgs_perceptual = Variable(x.type(Tensor))
            # Extraiem les característiques del batch d'entrada
            real_features = feature_extractor(real_imgs_perceptual)
            # Calculem la pèrdua de contingut
            perceptual_loss = args.content_weight * mse_loss(gen_features.relu2_2, real_features.relu2_2)

            # CALCULEM LA PÈRDUA DE DIVERSITAT
            # Afegim un soroll gaussià al batch d'entrada
            z = Variable(AddGaussianNoise(0,1).__call__(x).type(Tensor))
            # Creem un altre tensor amb un altre soroll gaussià al batch d'entrada
            z_2 = Variable(AddGaussianNoise(0,1).__call__(x).type(Tensor))
            # Generem un batch a partir del primer batch amb soroll
            fake_diversity = generator(z)
            # Generem un batch a partit del segon batch amb soroll
            fake_diversity_2 = generator(z_2)
            # Calculem la pèrdua de diversitat
            diversity_loss = 0.5 * torch.reciprocal(l1_loss(fake_diversity, fake_diversity_2))
            
            # # Definim els pesoso de les peerdues percetuals i de diversitat
            alpha = 1e-2
            beta = 1e-2
            
            # Sumem per obtenir la pèrdua total del generador
            loss_G = adversarial_loss + alpha*perceptual_loss + beta*diversity_loss
            
            # Computem els gradients
            loss_G.backward()
            optimizer_G.step()

            #################################
            #   Entrenem el Discriminador   #
            #################################
            
            # Inicialitzem l'optimitzador del generador
            optimizer_D.zero_grad()
    
            # Mesura de la habilitat del discriminador de classificar les imatges generades
            real_loss = mse_loss(discriminator(real_imgs), valid)
            fake_loss = mse_loss(discriminator(fake_imgs.detach()), fake)
            loss_D = 0.5 * (real_loss + fake_loss)

            # Computem els gradients
            loss_D.backward()
            optimizer_D.step()
            
            # Desem la historia de pèrdues
            d1_hist.append(loss_D)
            g1_hist.append(adversarial_loss)
            g2_hist.append(perceptual_loss)
            g3_hist.append(diversity_loss)
            g4_hist.append(loss_G)

            # Escrivim per consola l'epoch i el batch_id executats, i les pèrdues totals
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, batch_id, len(train_loader), loss_D.item(), loss_G.item())
            )
            
            # Desem la imatge generada cada interval de epoch especificat per paràmetre
            # batches_done = epoch * len(train_loader) + batch_id
            # if batches_done % args.log_interval == 0:
            #     save_image(fake_imgs.data[:25], "/content/drive/My Drive/TFM/Generative Adversarial network/Artsy-gan/images/%d.png" % batches_done , nrow=5, normalize=True)
            #     # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done , nrow=5, normalize=True)

        # S'actualitza la taxa d'aprenentatge
        lr_scheduler_G.step()
        lr_scheduler_D.step()

    # Desem el generador creat
    generator.eval().to(device)
    save_model_filename = "generator_epoch_" + str(args.n_epochs) + "_wchr.pth"
    save_model_path = os.path.join(args.save_model_dir, args.dataset_name, save_model_filename)
    torch.save(generator.state_dict(), save_model_path)
    
    # Desem el discriminador creat
    discriminator.eval().to(device)
    save_model_filename = "discriminator_epoch_" + str(args.n_epochs) + "_wchr.pth"
    save_model_path = os.path.join(args.save_model_dir, args.dataset_name, save_model_filename)
    torch.save(discriminator.state_dict(), save_model_path)
    
    # Grafiquem la història de pèrdues
    plot_history(d1_hist, g1_hist, g2_hist, g3_hist, g4_hist)
    
    # Final de l'entrenament
    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = GeneratorNet().to(device)
        state_dict = torch.load(args.model, map_location=device)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        
    utils.save_image(args.output_image, output[0])

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Artsy-GAN")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--n_epochs", type=int, default=200,
                                  help="number of training epochs, default is 200")
    train_arg_parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    train_arg_parser.add_argument("--batch-size", type=int, default=1,
                                  help="batch size for training, default is 1")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--dataset_name", type=str, required=True)
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=2e-4,
                                  help="learning rate, default is 2e-4")
    train_arg_parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    train_arg_parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    train_arg_parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    train_arg_parser.add_argument("--log-interval", type=int, default=5000,
                                  help="number of images after which the training loss is logged, default is 5000")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=5000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    train_arg_parser.add_argument("--content-weight", type=float, default=1,
                                  help="weight for content-loss, default is 1")

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
