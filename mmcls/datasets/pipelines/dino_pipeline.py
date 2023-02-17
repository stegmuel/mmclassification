from ..builder import PIPELINES
from torchvision import transforms as pth_transforms
from PIL import Image


@PIPELINES.register_module()
class DinoPipelineTrain(object):
    def __init__(self):
        self.transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, results):
        image = Image.fromarray(results['img'])
        results['img'] = self.transform(image)
        return results


@PIPELINES.register_module()
class DinoPipelineVal(object):
    def __init__(self):
        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, results):
        image = Image.fromarray(results['img'])
        results['img'] = self.transform(image)
        return results
