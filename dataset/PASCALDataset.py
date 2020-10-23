from .baseDataset import BaseDataset
class PASCALDataset(BaseDataset):
    def __init__(self, scale, imgPath = "./data/pascal/imgs/", edgePath = "./data/pascal/edges/"):
        super(PASCALDataset, self).__init__(imgPath, edgePath, scale, mask_suffix="")