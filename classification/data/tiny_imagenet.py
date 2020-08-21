import os
import warnings
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
# from IPython import embed

def get_loader(args, kwargs):
    warnings.filterwarnings('ignore')
    norm_mean = [0.4802, 0.4481, 0.3975]
    norm_std = [0.2302, 0.2265, 0.2262]
    loader_train = None

    if not args.test_only:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.remove(transform_list[0])
        
        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.ImageFolder(
                root=os.path.join(args.dir_data, 'tiny-imagenet-200', 'train'),
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        # embed()

    batch_test = 128
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize(norm_mean, norm_std)]
    transform_test = transforms.Compose(transform_list)

    loader_test = DataLoader(
        datasets.ImageFolder(
            root=os.path.join(args.dir_data, 'tiny-imagenet-200', 'val'),
            transform=transform_test),
        batch_size=batch_test, shuffle=False, **kwargs
    )

    return loader_train, loader_test
