from .baseDataset import BaseDataset
class CarvanaDataset(BaseDataset):
    def __init__(self, scale):
        super(CarvanaDataset, self).__init__("./data/carvana/imgs/", "./data/carvana/masks/", scale, mask_suffix="_mask")