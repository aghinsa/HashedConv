import copy
import torch
import numpy as np
import torch.nn as nn

from tqdm.auto import tqdm
from itertools import chain
from functools import reduce,partial

from base import *


def getattr_by_dot_path_list(obj,qual_path):
    """
    Parameters:
        obj: Python object
        qual_path: List[str]:Path by which obj will be dot indexed too
    Returns:
        obj.qp[0].qp[1]
    """
    return reduce(getattr,[obj]+ qual_path)

def getattr_by_path_list(obj,path_list):
    """
    Parameters:
        obj: Python object
        path_list: List of list of paths:

    Returns:
        List of attributes

        Eg:

        ```
        >>> class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
                    self.classifier = nn.Sequential(
                        nn.Dropout(),
                        nn.Linear(256 * 2 * 2, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                    )
        >>> model = FakeModel()
        >>> getattr_by_path_list(
                model,
                [
                    [["conv1"]],
                    [['classifier'], 1],
                    [['classifier'], 4],
                ]
            )

        [
            model.conv1,
            model.classifier["1"],
            model.classifier["4"],

        ]
        ```


    """
    def f(x,p):
        if isinstance(p,list):
            return getattr_by_dot_path_list(x,p)
        else:
            return x[p]
    r = [obj] + path_list
    attr = reduce(f,r)
    return attr


# getting paths to conv and linear
def get_layers_path(model):
    """
    Parameters:
        model: torch.model
    Returns:
        paths as list of lists for layers which has weight attribute

        Eg:

        ```
            >>> class FakeModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
                        self.classifier = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(256 * 2 * 2, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                        )
            >>> get_layers_path(FakeModel())

            [
                [["conv1"]],
                [['classifier'], 1],
                [['classifier'], 4],
            ]
        ```
    """
    all_paths = []
    for name,param in model.named_parameters():
        if "weight" in name: # classifier.1.weight
            qual_name = name.split('.')[:-1] # ['classifier','1']
            qual_path = []

            has_integer = False
            for x in qual_name:
                if x.isnumeric():
                    has_integer = True
                    x = int(x)
                qual_path.append(x)

            # if there are no integers then the path points directly to a layer
            if has_integer:
                new_qual_path = []
                t_path = []
                for x in qual_path:
                    if isinstance(x,int):
                        new_qual_path.append(t_path)
                        new_qual_path.append(x)
                        t_path = []
                        continue
                    t_path.append(x)
            else:
                new_qual_path = [qual_path]
            all_paths.append(new_qual_path)
    return all_paths

class BitQuantizer:

    def __init__(self,n_fs,n_bits,layer_names,):
        self.n_fs = n_fs
        self.n_bits = n_bits
        self.layer_names = layer_names
        for layer_name in layer_names:
            setattr(self,layer_name,LayerData())
        self.all_encodings = self.get_binary_encodings(self.n_bits)
        self.all_encodings = torch.from_numpy(self.all_encodings).float().cuda()

    def _init_hashed_coefs(self,w,n):
        hist,bin_edges = np.histogram(w,bins=64,density = True)
        vals = list(zip(hist,bin_edges[1:]))
        vals.sort(key = lambda x:x[0],reverse=True)
        vals = [ x[1] for x in vals[:6]  ]
        return np.array(vals).reshape(-1,1)

    def init_hash_functions(self,model):
        for layer in self.layer_names:
            w = getattr(model,layer).weight.data
            coefs = self._init_hashed_coefs(w.cpu(),self.n_bits)
            getattr(self,layer).fs = torch.from_numpy(np.repeat(coefs.reshape(1,-1),self.n_fs,axis=0)).cuda()
            getattr(self,layer).w = w
            getattr(self,layer).n_out = w.size()[0]


    def get_binary_encodings(self,n):
        """
        Generates all permutations of 1,-1 of given length n
        """
        ans = []
        low = (1<<n)
        high = (low << 1)

        for tx in range(low,high):
            tans = [-1]*n
            for j in range(n):
                if tx & (1<<j) :
                    tans[j]=1
            ans.append(tans)
        return np.array(ans)

    def train_hash_functions_for_layer(self,layer_name,n_iter = 100):
        layer_data = getattr(self,layer_name)
        w_master = layer_data.w

        with torch.no_grad():
            with torch.cuda.device("gpu"):
                for _ in tqdm(range(n_iter)):
                    channel_step = layer_data.n_out//self.n_fs

                    new_fs = []
                    new_ws = []

                    for fidx in range(0,self.n_fs):
                        # select hash functions for the channel batch
                        f = layer_data.fs[fidx].reshape(-1,1) #[nb,1]

                        if fidx != self.n_fs-1:
                            wx = w_master[fidx*channel_step:(fidx+1)*channel_step,:,:,: ] # [channel_step,inc,k,k]
                        else:
                            wx = w_master[fidx*channel_step:,:,:,: ] # [channel_step,inc,k,k]

                        # finding encoding for qhich quant level is closest to w
                        w = wx.reshape(-1,1)
                        quant_levels = torch.matmul(self.all_encodings, f) # [2^nb,1]

                        w = torch.abs( w - quant_levels.t()) # [m,2^nb]

                        idx = torch.argmin(w,dim = -1)

                        selected_encoding = self.all_encodings[idx] #[m,nbins]

                        # From the selected encodings generate hash values
                        new_w = torch.matmul(selected_encoding,f).reshape(wx.size())
                        new_ws.append(new_w)

                        # optimizing coefficients which minimizes square loss with
                        # current unhashed weights

                        s = selected_encoding #[m,nb]
                        psued_inv = torch.pinverse(s)
                        f_new_bin = torch.matmul(psued_inv,wx.reshape(-1,1)) #[nb,1]
                        new_fs.append(f_new_bin.t())

                    new_fs = torch.cat(new_fs,axis=0)
                    alpha = 0.1
                    layer_data.fs = layer_data.fs + alpha * (new_fs - layer_data.fs)


            hashed_weight = torch.cat(new_ws,axis=0)
            layer_data.hashed_weight = hashed_weight

        setattr(self,layer_name,layer_data)
        return

    def train_hash_functions(self,layer_names= None,n_iter = 100):
        if layer_names is None:
            layer_names = self.layer_names
        for layer_name in self.layer_names:
            print(f"hashing layer: {layer_name}")
            self.train_hash_functions_for_layer(layer_name,n_iter)

    def hash_model_layers(self,input_model,layer_names = None):
        model = copy.deepcopy(input_model)

        if layer_names is None:
            layer_names = self.layer_names
        for layer_name in layer_names:
            getattr(model,layer_name).weight = nn.Parameter(
                getattr(self,layer_name).hashed_weight)
        return model

