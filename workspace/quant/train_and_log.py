import sys
sys.path.insert(0,".")

import torch
import mlflow
import dill

from tqdm.auto import tqdm
from functools import partial
from itertools import product
from collections import namedtuple

from alexnet import AlexNet
from quantizer import BitQuantizer
from utils import cifar10_loader,evaluate
from pytorch_resnet.resnet import resnet20,resnet32,resnet44,resnet56,resnet110

TrainConfig = namedtuple("TrainConfig",["n_functions","n_bits","n_iter","model_name"])
VERBOSE = 0

def load_resnet(name):
    checkpoint = torch.load(f"./pytorch_resnet/pretrained_models/{name}.th")
    state = { k.replace('module.',''):v for k,v in checkpoint['state_dict'].items() }
    return state


def train_quantizer(model,train_config,n_epochs = 1000):
    try:
        if VERBOSE == 1:
            print(f"Training Config : {train_config}")
        model.cuda()
        trainloader,testloader = cifar10_loader(4096,"../data")
        n_fs = train_config.n_functions
        n_bits = train_config.n_bits
        n_iter = train_config.n_iter

        avoid =[]
        bit_quantizer = BitQuantizer(
                            model,
                            n_fs,
                            n_bits,
                            verbose = VERBOSE,
                            avoid=avoid
                    )
        with mlflow.start_run(run_name = str(train_config)):

            for epoch in tqdm(range(1,n_epochs+1)):

                bit_quantizer.train_hash_functions(n_iter = 1)
                hashed_model = bit_quantizer.get_hashed_model()
                hashed_model.cuda()


                # Evaluating hashed model
                train_acc_hashed = evaluate(hashed_model,trainloader,cuda = True)
                test_acc_hashed = evaluate(hashed_model,testloader,cuda = True)

                mlflow.log_metric("train_acc",train_acc_hashed,step = epoch)
                mlflow.log_metric("test_acc",test_acc_hashed,step = epoch)

        return hashed_model,bit_quantizer
    except RuntimeError as e:
        if 'out of memory' in str(e) :
            print(f'| WARNING: ran out of memory,skipping {train_config}')
            torch.cuda.empty_cache()


if __name__ == "__main__":

    # models
    # model_names = ["AlexNet","resnet20","resnet32","resnet44","resnet56","resnet110"]
    # models = [AlexNet,resnet20,resnet32,resnet44,resnet56,resnet110]
    # load_fns = [
    #     partial(torch.load ,"./alexnet_pretrained"),
    #     partial(load_resnet,"resnet20"),
    #     partial(load_resnet,"resnet32"),
    #     partial(load_resnet,"resnet44"),
    #     partial(load_resnet,"resnet56"),
    #     partial(load_resnet,"resnet110")
    # ]

    model_names = ["resnet20","resnet32","resnet44"]
    models = [resnet20,resnet32,resnet44]
    load_fns = [
        partial(load_resnet,"resnet20"),
        partial(load_resnet,"resnet32"),
        partial(load_resnet,"resnet44"),
    ]
    train_configs = [
        TrainConfig(
                n_functions = 64,
                n_bits = 6,
                n_iter = 1,
                model_name = "resnet20"
            ),
        TrainConfig(
                n_functions = 64,
                n_bits = 6,
                n_iter = 1,
                model_name = "resnet32"
            ),
        TrainConfig(
                n_functions = 64,
                n_bits = 6,
                n_iter = 1,
                model_name = "resnet44"
            ),
    ]


    for model_name,model_fn,ckpt_fn,config in zip(model_names,models,load_fns,train_configs):

        model = model_fn()
        model.load_state_dict(ckpt_fn())

        print(f"\nModel: {model_name} \n")

        hashed_model,bit_quantizer = train_quantizer(model,config,n_epochs = 1)

        with open(f"{model_name}_quantizer",'wb+') as f:
            dill.dump(bit_quantizer,f)









