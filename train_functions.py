import os
from itertools import cycle

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as D
import torchvision as tv


def train_fs_epoch(vae, vae_optim, vae_loss_fn,
                   classifier, classifier_optim,
                   classifier_loss_fn, use_cuda, split_early,
                   train_loader=None, alpha=1):
    """
    FULLY SUPERVISED TRAINING FUNCTION USING POSE VAE
    train vae encoder and classifier for one epoch
    returns loss for one epoch
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    num_steps = 0
    for data in train_loader:
        x = data['image']
        y = data['data']

        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # step of elbo for vae
        classifier_optim.zero_grad()
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, x)
        out, split = vae.encoder(x)
        if split_early:
            to_classifier = split
        else:
            z_dist = D.Normal(out["z_mu"], out["z_std"])
            to_classifier = z_dist.rsample()
        y_out = classifier.forward(to_classifier)
        classifier_loss = classifier_loss_fn(y_out, y)
        # step through classifier
        total_loss = vae_loss + alpha*classifier_loss
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        total_loss.backward()
        vae_optim.step()
        classifier_optim.step()
        num_steps += 1
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1), y.argmax(dim=1)))
    normalizer = len(train_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_acc_norm = total_acc /normalizer
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_acc_norm, num_steps


def train_ss_epoch(vae, vae_optim, vae_loss_fn,
                   classifier, classifier_optim,
                   classifier_loss_fn, use_cuda, split_early,
                   train_s_loader=None,
                   train_us_loader=None):
    """
    train vae and classifier for one epoch
    returns loss for one epoch
    in each batch, when the svi takes a step, the optimiser of
    classifier takes a step
    """
    # classifier is in train mode for dropout
    classifier.train()
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    num_steps = 0
    supervised_len = len(train_s_loader)
    unsupervised_len = len(train_us_loader)
    zip_list = zip(train_s_loader, cycle(train_us_loader)) if len(train_s_loader) > len(train_us_loader) else zip(cycle(train_s_loader), train_us_loader)
    for data_sup, data_unsup in zip_list:
        xs = data_sup['image']
        ys = data_sup['data']
        xus = data_unsup['image']
        if use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            xus = xus.cuda()
        classifier_optim.zero_grad()
        vae_optim.zero_grad()
        # supervised step
        vae_loss = vae_loss_fn(vae.model, vae.guide, xs)
        out, split = vae.encoder(xs)
        if split_early:
            to_classifier = split
        else:
            z_dist = D.Normal(out["z_mu"], out["z_std"])
            to_classifier = z_dist.rsample()

        y_out = classifier.forward(to_classifier)

        classifier_loss = classifier_loss_fn(y_out, ys)

        total_loss = vae_loss + classifier_loss
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1), ys.argmax(dim=1)))
        total_loss.backward()

        vae_optim.step()
        classifier_optim.step()

        # unsupervised step
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, xus)
        vae_loss.backward()
        vae_optim.step()
        num_steps += 1
        epoch_loss_vae += vae_loss.item()
    if supervised_len > unsupervised_len:
        normaliser = len(train_s_loader.dataset)
    else:
        normaliser = len(train_us_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / 2*normaliser
    total_epoch_loss_classifier = epoch_loss_classifier / normaliser
    total_acc_norm = total_acc / normaliser
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_acc_norm, num_steps


 def train_ss_bayes(vae, vae_optim, vae_loss_fn,
                   classifier, classifier_optim,
                   classifier_loss_fn, use_cuda, split_early, guide=None,
                   train_s_loader=None,
                   train_us_loader=None):
    """
    train vae and classifier for one epoch
    returns loss for one epoch
    in each batch, when the svi takes a step, the optimiser of
    classifier takes a step
    """
    # classifier is in train mode for dropout
    classifier.train()
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    num_steps = 0
    supervised_len = len(train_s_loader)
    unsupervised_len = len(train_us_loader)
    zip_list = zip(train_s_loader, cycle(train_us_loader)) if len(train_s_loader) > len(train_us_loader) else zip(cycle(train_s_loader), train_us_loader)
    for data_sup, data_unsup in zip_list:
        xs = data_sup['image']
        ys = data_sup['data']
        xus = data_unsup['image']
        if use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            xus = xus.cuda()
        classifier_optim.zero_grad()
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, xs)
        out, split = vae.encoder(xs)
        if split_early:
            to_classifier = split
        else:
            z_dist = D.Normal(out["z_mu"], out["z_std"])
            to_classifier = z_dist.rsample()

        y_out = classifier.forward(to_classifier)

        classifier_loss = classifier_loss_fn(classifier, guide, y_out, ys)

        total_loss = vae_loss + classifier_loss
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1), ys.argmax(dim=1)))
        total_loss.backward()

        vae_optim.step()
        classifier_optim.step()

        # unsupervised step
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, xus)
        vae_loss.backward()
        vae_optim.step()
        num_steps += 1
        epoch_loss_vae += vae_loss.item()
    if supervised_len > unsupervised_len:
        normaliser = len(train_s_loader.dataset)
    else:
        normaliser = len(train_us_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / 2*normaliser
    total_epoch_loss_classifier = epoch_loss_classifier / normaliser
    total_acc_norm = total_acc / normaliser
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_acc_norm, num_steps


def rms_calc(probs, target):
    """
    total rms for a single batch
    used in eval function
    """
    target = target.cpu().numpy()
    probs = probs.detach().cpu().numpy()
    total_count = np.sum(target, axis=1)
    probs_target = target / total_count[:, None]
    rms =  np.sqrt((probs - probs_target)**2)
    return np.sum(rms)


def evaluate(vae, vae_loss_fn, classifier,
             classifier_loss_fn, test_loader, use_cuda, transform=False, split_early=False):
    """
    evaluates for all test data
    test data is in batches, all batches in test loader tested
    """
    # classifier is in eval mode
    num_samples = 100
    classifier.eval()
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    rms = 0.
    for data in test_loader:
        x = data['image']
        y = data['data']
        if transform is not False:
            x = transform(x)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        out, split = vae.encoder(x)
        if split_early:
            to_classifier = split
            y_out = classifier.forward(to_classifier)

        else:
            z_dist = D.Normal(out["z_mu"], out["z_std"])
            to_classifier = z_dist.rsample([num_samples])
            y_out = classifier.forward(to_classifier)
            y_out = torch.mean(y_out, 0)
        vae_loss = vae_loss_fn(vae.model, vae.guide, x)
        if split_early:
            to_classifier = split
        else:
            z_dist = D.Normal(out["z_mu"], out["z_std"])
            to_classifier = z_dist.rsample()
        y_out = classifier.forward(to_classifier)
        classifier_loss = classifier_loss_fn(y_out, y)
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1), y.argmax(dim=1)))
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        rms += rms_calc(y_out, y)
    normalizer = len(test_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_epoch_acc = total_acc / normalizer
    rms_epoch = rms / normalizer
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc, rms_epoch


def train_log(train_fn,
              vae, vae_optim, vae_loss_fn,
              classifier, classifier_optim,
              classifier_loss_fn, dir_name, num_epochs,
              use_cuda, test_loader, split_early,
              train_fn_kwargs, bayesian=False,
              plot_img_freq=20, num_img_plt=9,
              checkpoint_freq=20,
              test_freq=1, transform=False):
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    writer = SummaryWriter("tb_data/" + dir_name)
    total_steps = 0
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
    if use_cuda:
        classifier.cuda()
    for epoch in range(num_epochs):
        print("training")
        if bayesian == True:
            svi = SVI(classifier, guide, classifier_optim, classifier_loss_fn)
            total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc, num_steps = train_fn()
        else:
            total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc, num_steps = train_fn(
            vae, vae_optim, vae_loss_fn,
            classifier, classifier_optim, classifier_loss_fn, use_cuda, split_early, **train_fn_kwargs)
        total_steps += num_steps
        print("end train")
        print("[epoch %03d]  average training loss vae: %.4f" % (epoch, total_epoch_loss_vae))
        print("[epoch %03d]  average training loss classifier: %.4f" % (epoch, total_epoch_loss_classifier))
        print("[epoch %03d]  average training accuracy: %.4f" % (epoch, total_epoch_acc))

        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test_vae, total_epoch_loss_test_classifier, accuracy, rms = evaluate(
                vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader, 
                use_cuda, transform=transform, split_early=split_early)
            print("[epoch %03d] average test loss vae: %.4f" % (epoch, total_epoch_loss_test_vae))
            print("[epoch %03d] average test loss classifier: %.4f" % (epoch, total_epoch_loss_test_classifier))
            print("[epoch %03d] average test accuracy: %.4f" % (epoch, accuracy))
            print("evaluate end")
            writer.add_scalar('Train loss vae', total_epoch_loss_vae, total_steps)
            writer.add_scalar('Train loss classifier', total_epoch_loss_classifier, total_steps)
            writer.add_scalar('Train accuracy', total_epoch_acc, total_steps)
            writer.add_scalar('Test loss vae', total_epoch_loss_test_vae, total_steps)
            writer.add_scalar('Test loss classifier', total_epoch_loss_test_classifier, total_steps)
            writer.add_scalar('Test accuracy', accuracy, total_steps)
            writer.add_scalar('rms normalised', rms, total_steps)

        if epoch % plot_img_freq == 0:
            image_in = next(iter(test_loader))['image'][0:num_img_plt]
            images_out = vae.sample_img(image_in, use_cuda)
            img_grid_in = tv.utils.make_grid(image_in)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images in, from step' + str(total_steps), img_grid_in)
            writer.add_image(str(num_params) + ' images out, from step'+ str(total_steps), img_grid)

        if epoch % checkpoint_freq == 0:

            torch.save(vae.encoder.state_dict(), "checkpoints/" + dir_name + "/encoder.checkpoint")
            torch.save(vae.decoder.state_dict(),  "checkpoints/" + dir_name +  "/decoder.checkpoint")
            torch.save(classifier.state_dict(),  "checkpoints/" + dir_name +  "/classfier.checkpoint")

        writer.close()
