from .baseDataset import BaseDataset
class MappingChallengeDataset(BaseDataset):
    def __init__(self, scale, isEdge=False):
        maskDir = "./data/mapping-challenge/edges/" if isEdge else "./data/mapping-challenge/masks/"
        super(MappingChallengeDataset, self).__init__("./data/mapping-challenge/images/", maskDir, scale)