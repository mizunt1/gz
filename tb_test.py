from load_mnist import setup_data_loaders
from torch.utils.tensorboard import SummaryWriter
import torchvision
def reconstruct_img(x):
    # encode image x
    z_loc, z_scale = self.encoder(x)
    # sample in latent space
    z = dist.Normal(z_loc, z_scale).sample()
    # decode the image (note we don't sample in image space)
    loc_img = self.decoder(z)
    return loc_img

writer = SummaryWriter("tb_data/")
train_loader, test_loader = setup_data_loaders(batch_size=9)
images, labels = next(iter(train_loader))
img_grid = torchvision.utils.make_grid(images)
writer.add_image('images', img_grid)
writer.close()
