class AttrGetter:
    def __init__(self):
        pass
    def get(self,obj,path):
        """
        Path : Dot seperated path to attribute
        """
        pass
    def overwrite(self,obj,path,val):
        pass

class LayerData:
    def __init__(self,qual_path):
        self.qual_path = qual_path
        self.fs = None
        self.w = None
        self.n_out = None
        self.hashed_weight = None
        self.n_fs = None

        f = lambda L: '.'.join([f(x) if type(x) is list else x for x in L])
        self.layer_name = f([str(x) if isinstance(x,int) else x for x in self.qual_path])

    def get_layer(self,model,path):
        pass

