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
    def __init__(self):
        self.qual_path = None
        self.fs = None
        self.w = None
        self.n_out = None
        self.hashed_weight = None