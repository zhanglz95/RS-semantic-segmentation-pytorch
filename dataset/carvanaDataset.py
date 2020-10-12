from .baseDataset import BaseDataset
class CarvanaDataset(BaseDataset):
    def __init__(self, scale, isEdge=False):
        maskDir = "./data/carvana/edges/" if isEdge else "./data/carvana/masks/"
        suffix = "_edge" if isEdge else "_mask"
        super(CarvanaDataset, self).__init__("./data/carvana/imgs/", maskDir, scale, mask_suffix=suffix)