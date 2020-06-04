import torch
import pyro
from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, Categorical
from pyro.infer import Predictive

class ClassifierBnn(PyroModule):
    
    def __init__(self, num_in=784, num_hidden = 200, num_out = 10, prior_std = 1.):
        
        # call to father constructor
        super().__init__()
        
        # define prior
        prior = Normal(0, prior_std)
        
        # Define layers
        
        # linear layer 1
        self.linear_layer = PyroModule[torch.nn.Linear](num_in, num_hidden)
        
        # linear alyer parameters as random variables
        self.linear_layer.weights = PyroSample(prior.expand([num_hidden, num_in]).to_event(2))
        self.linear_layer.bias = PyroSample(prior.expand([num_hidden]).to_event(1))
        
        # linear layer 2
        # output dimension is 3 because of the number of classes
        self.output_layer = PyroModule[torch.nn.Linear](num_hidden, num_out)
        
        # linear alyer parameters as random variables
        self.output_layer.weights = PyroSample(prior.expand([num_out, num_hidden]).to_event(2))
        self.output_layer.bias = PyroSample(prior.expand([num_out]).to_event(1))
        
        # activation function
        #self.activation = torch.nn.functional.softmax()
        
    def forward(self, x, y = None):
        # latent variable
        z = self.linear_layer(x)
        z = self.output_layer(z)
        z = torch.nn.functional.log_softmax(z, dim=1)
        # likelihood
        with pyro.plate("data",size = x.shape[0], dim = -1):
            # I think this means each batch is independent            
            # z is the input to the distribution (categorical)
            # sample an observation obs depending on categorial probabilities z
            # z is vector of probabilities, size [batch_size, 10]
            obs = pyro.sample("obs", Categorical(logits = z), obs=y)
        # return latent variables
        # when model is sampled, is Z returned, or a single category? Not sure
        return z

def validate_model(train_loader, encoder=False, transform=False):
    pyro.enable_validation(True)
    model = ClassifierBnn()
    x, y = next(iter(train_loader))
    if transform != False:
        x = transform(x)
    if encoder != False:
        z_loc, z_scale = encoder(x)
        x = torch.cat((z_loc, z_scale), 1)
    print(pyro.poutine.trace(model).get_trace(x, y).format_shapes())
    Model = ClassifierBnn()
    pyro.enable_validation(True)
    pyro.clear_param_store()
    model = ClassifierBnn(num_hidden = 10, prior_std = 1.)



def predict(x,model, guide, transform=False):
    predictive = Predictive(model, guide=guide, num_samples=20)
    if transform != False:
        x = transform(x)
    # for a single image, output a mean and sd for category
    yhats = predictive(x)["obs"].double()
    # yhats[0] seems to be integers 0 to 9, len 256
    # prediction for one model, for all items in batch
    #yhats shape is 20, 256
    # predictions for each item in batch and each model
    # take a mean across 20 models
    # Doesnt make sense to take mean, should take mode
    mode = torch.mode(yhats, axis=0)

    std = torch.std(yhats.float(), 0).numpy()
    # yhats outputs a batch size number of predictions for 20 models
    # yhats seem to be a dictionary of weights
    return mode

def evaluate_test(test_loader, model, guide, encoder=False, transform=False):
    accuracy = 0
    for x, y in test_loader:
        if transform != False:
            x = transform(x)
        if encoder != False:
            z_loc, z_scale = encoder(x)
            x = torch.cat((z_loc, z_scale), 1)
        mode = predict(x, model, guide, transform=transform)
        num_correct_in_batch = torch.sum(torch.eq(mode.values,y))
        accuracy += num_correct_in_batch.numpy()/len(y)
    return accuracy / (len(test_loader))
        
num_epochs = 1000


test_freq = 10

def train_bnn(num_epochs, train_loader, test_loader, model, guide, encoder=False, transform=False):
    svi = pyro.infer.SVI(model,
                    guide,
                    optim=pyro.optim.ClippedAdam({'lr':1e-3}),
                    # Define conventional ELBO
                     loss=pyro.infer.Trace_ELBO())

    for epoch in range(num_epochs):
        i = 0
        for x, y in train_loader:
            if transform != False:
                x = transform(x)
            i +=1
            # batches of size 256 are being fed in 
            if encoder != False:
                z_loc, z_scale = encoder(x)
                combined_z = torch.cat((z_loc, z_scale), 1)
                x = combined_z
            loss = svi.step(x, y)
            if i % test_freq == 0:
                test_acc = evaluate_test(test_loader, model, guide, encoder, transform=transform)
                print("test acc", test_acc)
                print("train acc", accuracy_per_batch)
                print("loss", loss)

            mean, std = predict(x, model, guide)
            accuracy_per_batch = torch.sum(torch.eq(mean.int(),y)).numpy()/len(y)
