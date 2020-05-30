import sys

sys.path.insert(0, ".")

import torch
import mlflow

from tqdm.auto import tqdm
from functools import partial
from itertools import product
from collections import namedtuple

from alexnet import AlexNet
from quantizer import BitQuantizer
from utils import cifar10_loader, evaluate
from pytorch_resnet.resnet import resnet20, resnet32, resnet44, resnet56, resnet110

TrainConfig = namedtuple(
    "TrainConfig", ["n_functions", "n_bits", "n_iter", "model_name"]
)


def load_resnet(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    return state


def train_model_hash(model, train_config):
    try:
        if VERBOSE == 1:
            print(f"Training Config : {train_config}")
        model.cuda()
        trainloader, testloader = cifar10_loader(4096, "./data")
        n_fs = train_config.n_functions
        n_bits = train_config.n_bits
        n_iter = train_config.n_iter

        avoid = []
        bit_quantizer = BitQuantizer(model, n_fs, n_bits, verbose=VERBOSE, avoid=avoid)
        bit_quantizer.train_hash_functions(n_iter=n_iter)
        hashed_model = bit_quantizer.get_hashed_model()
        hashed_model.cuda()

        # Evaluating model before hashing
        train_acc = evaluate(model, trainloader, cuda=True)
        test_acc = evaluate(model, testloader, cuda=True)

        # Evaluating hashed model
        train_acc_hashed = evaluate(hashed_model, trainloader, cuda=True)
        test_acc_hashed = evaluate(hashed_model, testloader, cuda=True)

        with mlflow.start_run(run_name=str(train_config)):
            mlflow.log_param("train_acc_before_hashing", train_acc)
            mlflow.log_param("test_acc_before_hashing", test_acc)
            mlflow.log_param("train_acc_after_hashing", train_acc_hashed)
            mlflow.log_param("test_acc_after_hashing", test_acc_hashed)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_bits", n_bits)
            mlflow.log_param("n_fs", n_fs)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"| WARNING: ran out of memory,skipping {train_config}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    VERBOSE = 0

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

    model_names = ["resnet20", "resnet32", "resnet44"]
    models = [resnet20, resnet32, resnet44]
    fp = lambda name: f"./pytorch_resnet/pretrained_models/{name}.th"
    load_fns = [
        partial(load_resnet, fp("resnet20")),
        partial(load_resnet, fp("resnet32")),
        partial(load_resnet, fp("resnet44")),
    ]

    # parameters
    n_functions = [1, 2, 4, 8, 16, 32, 64]
    n_bits = [1, 2, 3, 4, 5, 6]

    # Fixing niter
    n_iter = 15

    for model_name, model_fn, ckpt_fn in zip(model_names, models, load_fns):

        model = model_fn()
        model.load_state_dict(ckpt_fn())

        fbs = list(product(n_functions, n_bits))
        print(f"\nModel: {model_name} \n")
        for n_f, n_b in tqdm(fbs):
            config = TrainConfig(
                n_functions=n_f, n_bits=n_b, n_iter=n_iter, model_name=model_name
            )
            train_model_hash(model, config)
