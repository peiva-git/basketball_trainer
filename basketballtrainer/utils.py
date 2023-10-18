import paddle as pp

from PIL import Image


def visualize_random_crops(crops: list[pp.Tensor]):
    for counter, crop in enumerate(crops):
        crop = pp.squeeze(crop)
        crop = pp.moveaxis(crop, source=0, destination=-1)
        crop = pp.cast(crop, pp.uint8)
        Image.fromarray(crop.numpy()).show()
