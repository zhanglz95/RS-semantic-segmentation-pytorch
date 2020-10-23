from .baseDataset import BaseDataset
class VaihingenDataset(BaseDataset):
    def __init__(self, scale, imgPath = "./data/vaihingen/imgs/", maskPath = "./data/vaihingen/masks/"):
        super(VaihingenDataset, self).__init__(imgPath, maskPath, scale)