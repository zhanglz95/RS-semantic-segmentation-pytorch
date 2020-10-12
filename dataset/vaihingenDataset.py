from .baseDataset import BaseDataset
class VaihingenDataset(BaseDataset):
    def __init__(self, scale, isEdge=False):
        maskDir = "./data/vaihingen/edges/" if isEdge else "./data/vaihingen/masks/"
        super(VaihingenDataset, self).__init__("./data/vaihingen/imgs/", maskDir, scale)