import paddle as pp


class PaddleSegDatasetBuilder(pp.io.Dataset):
    def __init__(self):
        super(PaddleSegDatasetBuilder, self).__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
