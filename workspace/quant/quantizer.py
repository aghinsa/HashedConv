import copy
import torch
import types
import pickle
import inspect
import numpy as np
import torch.nn as nn


from tqdm.auto import tqdm
from itertools import chain
from functools import reduce,partial

from base import *

import numpy as np


def get_binary_encodings(n):
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

def setattr_by_path_list(obj,input_path_list,val):
    """
    Set value of attribute indexed by path list to val

    Parameters:
        obj : Python object
        input_path_list : List[List[Union[str,int]]
        val:Any

    Returns :
        None

    Eg:
        ```
        >>> class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
                    self.classifier = nn.Sequential(
                        nn.Linear(100, 1000),
                        nn.Linear(1000, 10),

                    )
        >>> model = FakeModel()
        >>> model.classifier[0]
        nn.Linear(100, 1000)

        >>> new_linear = nn.Linear(10, 10)
        >>> path_list = [['classifier'], 0]
        >>> setattr_by_path_list(model,path_list , new_linear )
        >>> model.classifier[0]

        nn.Linear(10,10)

        ```
    """
    path_list = copy.deepcopy(input_path_list)
    last_path = path_list.pop()

    def set_by_one(obj,l,val):
        if isinstance(l,list):
            t_obj = getattr_by_path_list(obj,l[:-1])
            set_by_one(t_obj,l[-1],val)
        elif isinstance(l,int):
            obj[l]=val
            return
        elif isinstance(l,str):
            setattr(obj,l,val)
            return

    # path list only had one element
    if not path_list:
        set_by_one(obj,last_path,val)
        return
    else:
        t_obj = getattr_by_path_list(obj,path_list)
        set_by_one(t_obj,last_path,val)

        return



# getting paths to all layers in model
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
    """Class to hash weights of a model to specified number of bins.

    Eg:
        ```
        >>> model = resnet32()
        >>> n_fs = 32
        >>>  n_bits = 6
        >>>  n_iter = 50
        >>>  avoid =[]
        >>>  bit_quantizer = BitQuantizer(
                                model,
                                n_fs,
                                n_bits,
                                avoid=avoid
                        )
        >>> bit_quantizer.train_hash_functions(n_iter = n_iter)
        >>> hashed_model = bit_quantizer.get_hashed_model()
        ```
    """
    def __init__(self,model,n_fs,n_bits,layers_qual_path=None,avoid=[],verbose=2):
        """
        Parameters:
            - model:torch.model : model to quantize.
            - n_fs:int : number of functions to use in layer,
                    minimum of n_fs,number of out channels will be used.
            - n_bits:int : number of bits per weight
            - layers_qual_path:List[List[str,int]] : path to layers which should be
                    quantized,should have weight as an attribute. All convolutional,
                    linear layers are used if not provided.Default:None.
            - avoid:List[str] : List of layers to be avoided.Layer name should be dot
                                seperated path.eg: ["classifier.1.conv1"].Default = [].
            - verbose:int : one of 0,1,2. 2 being the most detailed.
        """
        self.model = model
        self.n_fs = n_fs
        self.n_bits = n_bits
        self.layers_qual_path = layers_qual_path
        self.verbose = verbose

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


        if self.verbose == 2:
            print(f"Layers detected : {[x.layer_name for x in self.layer_datas ]}")

        self.all_encodings = get_binary_encodings(self.n_bits)
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

    def train_hash_functions_for_layer(self,layer_data,n_iter = 100):
        """
        Minimizes hash parameter for layer pointed by layer_data.layer_path
        Parameters:
            layer_data:LayerData
            n_iter:int
        Returns: None
        """
        w_master = layer_data.w
        with torch.no_grad():
            with torch.cuda.device("cuda"):
                disable_tqdm = (self.verbose < 2)
                for _ in tqdm(range(n_iter),disable=disable_tqdm):
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

        return

    def train_hash_functions(self,n_iter):
        """
        Trains hash functions for all layers

        Parameters:
            n_iter:int
        """
        disable_tqdm = self.verbose < 1
        for layer_data in tqdm(self.layer_datas,disable = disable_tqdm):

            if self.verbose == 1:
                print(f"Hashing layer: {layer_data.layer_name}")
            self.train_hash_functions_for_layer(layer_data,n_iter)

            if self.verbose == 1:
                print()

    def get_hashed_model(self):
        """
        Copies hashed weights to the model

        Return:
            torch.model
        """
        model = copy.deepcopy(self.model)
        for layer_data in self.layer_datas:
            getattr_by_path_list(model,layer_data.qual_path).weight = nn.Parameter(layer_data.hashed_weight)
        return model




class QuantConv2d(nn.Conv2d):
    """Convolutional layer which uses quantized weights.Can be used in place
    of nn.Conv2d
    """
    def __init__(self,*args,**kwargs):
        """
        Parameters:
            - All parameters supported by nn.Conv2d
            - n_bits:int : number of bits to quantize
            - n_functions:int :number of functions to use per layer
        """
        self.n_bits = 6
        self.n_fs = 64
        if "n_bits" in kwargs:
            self.n_bits = kwargs["n_bits"]
            kwargs.pop("n_bits")
        if "n_functions" in kwargs:
            self.n_fs = kwargs["n_functions"]
            kwargs.pop("n_functions")

        super().__init__(*args,**kwargs)


        # w size (outc,inc,k,k)
        self.n_out = self.weight.size()[0]
        self.n_fs = min(self.n_fs,self.n_out)

        self.f_bins = nn.Parameter(torch.rand(self.n_fs,self.n_bits,))

        # generate all possible encodings of length n_bits
        self.all_encodings = np.array(get_binary_encodings(self.n_bits))
        self.all_encodings = torch.from_numpy(self.all_encodings) # [2^n_bits,n_bits]
        self.all_encodings = self.all_encodings.float().cuda()

        self.weight_hashed = self.weight.data

        self.hash_loss = 0
        self.mse = nn.MSELoss(reduction = "mean")

        self.quant_parameters = []


    def forward(self,x):
        if self.training:
            new_fs = []
            new_ws = []
            quant_parameters = []

            channel_step = self.out_channels//self.n_fs
            w_master = self.weight.data.detach()
            self.hash_loss = 0

            for fidx in range(0,self.n_fs):
                # select hash functions for the channel batch
                f = self.f_bins[fidx].reshape(-1,1) #[nb,1]
                with torch.no_grad():


                    if fidx != self.n_fs-1:
                        wx = w_master[fidx*channel_step:(fidx+1)*channel_step] # [channel_step,inc,k,k]
                    else:
                        wx = w_master[fidx*channel_step:] # [channel_step,inc,k,k]

                    # finding encoding for qhich quant level is closest to w
                    w = wx.reshape(-1,1)
                    quant_levels = torch.matmul(self.all_encodings, f) # [2^nb,1]

                    w = torch.abs( w - quant_levels.t()) # [m,2^nb]

                    idx = torch.argmin(w,dim = -1)

                    selected_encoding = self.all_encodings[idx] #[m,nbins]


                # From the selected encodings generate hash values
                new_w = torch.matmul(selected_encoding,f).reshape(wx.size())
                self.hash_loss += self.mse(new_w,wx)
                new_ws.append(new_w)
                quant_parameters.append( (selected_encoding,f,wx.size()) )


            weight_hashed = torch.cat(new_ws,axis=0)
            self.weight_hashed = weight_hashed
            self.quant_parameters = quant_parameters

            out = self.conv2d_forward(x,self.weight)
            if self.bias is not None:
                out+= self.bias.unsqueeze(1).unsqueeze(1)
            return out

        else:
            out = self.conv2d_forward(x,self.weight_hashed)
            if self.bias is not None:
                out+= self.bias.unsqueeze(1).unsqueeze(1)
            return out

    def set_quant_parameters(self,quant_parameters):
        self.quant_parameters = quant_parameters

        ws = []
        for e,f,s in self.quant_parameters:
            w = torch.matmul(e,f).reshape(s)
            ws.append(w)

        weight_hashed = torch.cat(ws,axis=0).cuda()
        self.weight_hashed = weight_hashed
        self.weight = nn.Parameter(weight_hashed)



def use_hashed_conv(model,n_bits=6,n_functions=64):
    """
    Replaces Conv2d by QuantConv2d in all children modules

    Parameters:
        - model:torch model
        - n_bits:int
        - n_functions:int

    Returns
        - torch model
    """
    layer_paths = get_layers_path(model,avoid = [])
    layers = [getattr_by_path_list(model,layer_path) for layer_path in layer_paths ]

    conv_layers_with_path = [
        (layer,layer_path) for layer,layer_path in zip(layers,layer_paths)
            if isinstance(layer,nn.Conv2d)
    ]

    hashed_convs = []

    # iterating through conv layers and creating kwargs
    conv_attrs = [ param_name for param_name in inspect.signature(nn.Conv2d).parameters ]
    # bias will be later copied from conv layer
    # initializing bias directly after copying from init method
    # of original layer will lead to error as, in nn.Conv2d bias is later changed
    # according tovalue of parameter `bias`
    conv_attrs.remove("bias")

    for layer,layer_path in conv_layers_with_path:
        kwargs = {
            attr: copy.deepcopy(getattr(layer,attr))
                for attr in conv_attrs
        }
        kwargs["n_bits"] = n_bits
        kwargs["n_functions"] = n_functions

        hashed_conv = QuantConv2d(**kwargs)
        hashed_conv.weight = nn.Parameter(copy.deepcopy(layer.weight.clone().detach()))

        if layer.bias is not None:
            hashed_conv.bias = nn.Parameter(copy.deepcopy(layer.bias.clone().detach()))
        else:
            hashed_conv.bias = None

        hashed_convs.append(hashed_conv)

    # replacing conv layers with hasehd conv
    for hashed_conv,(_,layer_path) in zip(hashed_convs,conv_layers_with_path):
        setattr_by_path_list(model,layer_path,hashed_conv)
    return model


def quantize_model_instance(model,n_bits=6,n_functions=64):
    """
    Converts torch model to quantized model

    Parameters:
        - model:torch model
        - n_bits:int
        - n_functions:int

    Returns
        - torch model
    """
    model = use_hashed_conv(model,n_bits,n_functions)

    layer_paths = get_layers_path(model,avoid = [])
    layers = [getattr_by_path_list(model,layer_path) for layer_path in layer_paths ]

    conv_layers = [
        layer for layer in layers if isinstance(layer,nn.Conv2d)
    ]

    def get_hash_loss(self):
        loss = 0
        for layer in conv_layers:
            loss += layer.hash_loss
        return loss

    def copy_hashed_weights(self):
        for layer in conv_layers:
            layer.weight = nn.Parameter(layer.weight_hashed)
        return

    model.get_hash_loss = types.MethodType(get_hash_loss,model)
    model.copy_hashed_weights = types.MethodType(copy_hashed_weights,model)

    return model

def quantizeModel(n_bits=6,n_functions=64):
    """
    Eg:
        ```
         >>> @quantizeModel(n_functions = 4,n_bits=2)
             class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
                    self.encoder = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, ),
                        nn.Conv2d(64, 64, kernel_size=3, )
                    )
        >>>  model = FakeModel()
        >>> isinstance(model.encoder[0],QuantConv2d)
        True

        >>> class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
                    self.encoder = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, ),
                        nn.Conv2d(64, 64, kernel_size=3, )
                    )
        >>> model = FakeModel()
        >>> isinstance(model.encoder[0],QuantConv2d)
        False
        >>> model = quantizeModel()(model)
        >>> isinstance(model.encoder[0],QuantConv2d)
        >>> True

        ```
    """
    def _quantizeModel(model_def):
        if isinstance(model_def,nn.Module):
            return quantize_model_instance(model_def,n_bits,n_functions)

        if inspect.isclass(model_def):
            class QuantizedModelDef:
                def __new__(cls,*args,**kwargs):
                    model = model_def(*args,**kwargs)
                    model = quantize_model_instance(model,n_bits,n_functions)
                    return model
            return QuantizedModelDef

    return _quantizeModel




