from .baseDataset import BaseDataset
class VaihingenDataset(BaseDataset):
    def __init__(self, scale):
        super(VaihingenDataset, self).__init__("./data/vaihingen/imgs/", "./data/vaihingen/masks/", scale)