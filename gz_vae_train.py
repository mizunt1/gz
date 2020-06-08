from gz_vae_conv import VAE, train, evaluate
from load_gz_data import Gz2_data, return_data_loader
from simple_classifier import Classifier
import torch
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
data = Gz2_data(csv_dir="~/diss/gz2_data/gz_amended.csv",
                image_dir="/Users/Mizunt/diss/gz2_data",
                list_of_interest=[a01,
                                  a02,
                                  a03])


vae = VAE()

optimizer = Adam({"lr": 1.0e-3})

svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
train_elbo = []
test_elbo = []
USE_CUDA = False
TEST_FREQUENCY = 1

train_loader, test_loader = return_data_loader(data, 0.2, 20)


# training VAE

for epoch in range(10):
    print("training")
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    print("end train")
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        print("evaluating")
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
        print("evaluate end")
        #torch.save(vae.encoder.state_dict(), "encoder.checkpoint")
        #torch.save(vae.decoder.state_dict(), "decoder.checkpoint")
