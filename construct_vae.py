import os
import pyro
import torch
import torch.nn as nn
import torchvision as tv
import pyro.distributions as dist
from torch.utils.tensorboard import SummaryWriter

class VAE(nn.Module):
    def __init__(self, encoder, decoder, z_dim, kwargs_encoder, kwargs_decoder, use_cuda=False):
        super().__init__()
        self.encoder = encoder(**kwargs_encoder)
        self.decoder = decoder(**kwargs_decoder)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            # decoder is where the image goes
            # is this correct?
            # Ask lewis, channel, and h and w are dependent, go to event
            pyro.sample(
                "obs",
                dist.Laplace(loc_img, 0.5).to_event(3),
                obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale, split = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#            pyro.sample("latent", dist.Delta(z_loc).to_event(1))

    # define a helper function for reconstructing images
    def sample_img(self, x, use_cuda=False, encoder=False, decoder=False):
        # encode image x
        if use_cuda == True:
            x = x.cuda()
        batch_shape = x.shape[0]
        img_shape = x.shape[-1]
        if encoder == False:
            z_loc, z_scale, split = self.encoder(x)
        else:
            z_loc, z_scale, split = encoder(x)
        # sample in latent space
#        z = dist.Delta(z_loc).sample()
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        if decoder == False:
            loc_img = self.decoder(z)
        else:
            loc_img = decoder(z)
        return loc_img.reshape([batch_shape, 1, img_shape, img_shape])

def train(svi, train_loader, use_cuda=False, transform=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        x = x['image']
        if transform != False:
            x = transform(x)
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        x = x['image']
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def train_log_vae(checkpoint_dir, writer_name, vae, svi, train_loader, test_loader,
                  num_epochs, plot_img_freq=1, num_img_plt=40,
                  checkpoint_freq=1, use_cuda=True, test_freq=1):
    
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

    writer = SummaryWriter("tb_data_all/" + writer_name)
    if not os.path.exists("checkpoints/" + checkpoint_dir):
        os.makedirs("checkpoints/" + checkpoint_dir)
    for epoch in range(num_epochs):
        print("training")
        total_epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
        print("end train")
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            print("evaluate end")
            writer.add_scalar('Train loss', total_epoch_loss_train, epoch)
            writer.add_scalar('Test loss', total_epoch_loss_test, epoch)
            print(epoch)
        if epoch % plot_img_freq == 0:
            image_in = next(iter(train_loader))['image'][0:num_img_plt]
            images_out = vae.sample_img(image_in, use_cuda=use_cuda)
            img_grid_in = tv.utils.make_grid(image_in)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images in, from epoch' + str(epoch), img_grid_in)
            writer.add_image(str(num_params) + ' images out, from epoch'+ str(epoch), img_grid)

        if epoch % checkpoint_freq == 0:

            torch.save(vae.encoder.state_dict(), "checkpoints/" + checkpoint_dir + "/encoder.checkpoint")
            torch.save(vae.decoder.state_dict(),  "checkpoints/" + checkpoint_dir +  "/decoder.checkpoint")
            
        writer.close()
