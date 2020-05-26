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
def get_layers_path(model,avoid = []):
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

            t_name = '.'.join(qual_name)
            if t_name in avoid:
                continue

            qual_path = [ int(x) if x.isnumeric() else x for x in qual_name]

            # splitting list on integers
            new_qual_path = []
            while(qual_path):
                t_path = []
                while(qual_path):
                    x = qual_path.pop(0)
                    if isinstance(x,int):
                        if t_path:
                            new_qual_path.append(t_path)
                            t_path=[]
                            new_qual_path.append(x)
                            break
                    else:
                        t_path.append(x)
                if t_path:
                    new_qual_path.append(t_path)

            all_paths.append(new_qual_path)
    return all_paths

class BitQuantizer:

    def __init__(self,model,n_fs,n_bits,layers_qual_path=None,avoid=[]):
        self.model = model
        self.n_fs = n_fs
        self.n_bits = n_bits
        self.layers_qual_path = layers_qual_path

        if self.layers_qual_path is None:
            self.layers_qual_path = get_layers_path(model,avoid = avoid)

        self.layer_datas = []
        for layer_number,layer_path in enumerate(self.layers_qual_path):
            layer_data = LayerData(qual_path=layer_path)
            model_layer = getattr_by_path_list(self.model,layer_path)

            if not(isinstance(model_layer,nn.Conv2d) or isinstance(model_layer,nn.Linear)):
                continue

            layer_data.get_layer = partial(getattr_by_path_list,self.model,layer_path)
            self.layer_datas.append(layer_data)


        print(f"Layers detected : {[x.layer_name for x in self.layer_datas ]}")

        self.all_encodings = self.get_binary_encodings(self.n_bits)
        self.all_encodings = torch.from_numpy(self.all_encodings).float().cuda()

        self.init_hash_functions()

    def _init_hashed_coefs(self,w,n,bins=64):
        """
        Calculates histogram of w with bins and returns the
        top n bin edges with highest density

        Parameters:
            w: np array
            n: Top n bins to select
        Returns:
            np.array [n,1]

        """
        hist,bin_edges = np.histogram(w,bins=bins,density = True)
        vals = list(zip(hist,bin_edges[1:]))
        vals.sort(key = lambda x:x[0],reverse=True)
        vals = [ x[1] for x in vals[:n]  ]
        return np.array(vals).reshape(-1,1)

    def init_hash_functions(self):
        """
        Initializes coefficints of hashed functions to
        top `n_bits` bins in each layers pretrained weight.
        """
        for layer_data in self.layer_datas:
            w = layer_data.get_layer().weight.data
            coefs = self._init_hashed_coefs(w.cpu(),self.n_bits)
            layer_n_fs = min(self.n_fs,w.size()[0])
            layer_data.fs = torch.from_numpy(np.repeat(coefs.reshape(1,-1),layer_n_fs,axis=0)).cuda()
            layer_data.n_fs = layer_n_fs
            layer_data.w = w
            layer_data.n_out = w.size()[0]

    def get_binary_encodings(self,n):
        """
        Generates all permutations of 1,-1 of length n

        Eg:
            >>> get_binary_encodings(2)

            [ [-1,-1],[-1,1],[1,-1],[1,1] ]
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

    def train_hash_functions_for_layer(self,layer_data,n_iter = 100):
        w_master = layer_data.w

        with torch.no_grad():
            with torch.cuda.device("cuda"):
                for _ in tqdm(range(n_iter)):
                    channel_step = layer_data.n_out//layer_data.n_fs

                    new_fs = []
                    new_ws = []

                    for fidx in range(0,layer_data.n_fs):
                        # select hash functions for the channel batch
                        f = layer_data.fs[fidx].reshape(-1,1) #[nb,1]

                        if fidx != layer_data.n_fs-1:
                            wx = w_master[fidx*channel_step:(fidx+1)*channel_step ] # [channel_step,inc,k,k]
                        else:
                            wx = w_master[fidx*channel_step:] # [channel_step,inc,k,k]

                        # finding encoding for qhich quant level is closest to w
                        w = wx.reshape(-1,1)
                        quant_levels = torch.matmul(self.all_encodings, f) # [2^nb,1]

                        w = torch.abs( w - quant_levels.t()) # [m,2^nb]

                        if(w.size()[0] == 0):
                            raise Exception(f"Difference between quant_levels and weight is of shape {w.size()}."
                                        +" Pease verify that final layers are not quantized.")
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

        # Check if layer data is getting updates
        # setattr(self,layer_name,layer_data)
        return

    def train_hash_functions(self,n_iter):
        for layer_data in self.layer_datas:
            print(f"hashing layer: {layer_data.layer_name}")
            self.train_hash_functions_for_layer(layer_data,n_iter)

    def get_hashed_model(self):
        model = copy.deepcopy(self.model)

        for layer_data in tqdm(self.layer_datas):
            getattr_by_path_list(model,layer_data.qual_path).weight = nn.Parameter(layer_data.hashed_weight)
        return model

