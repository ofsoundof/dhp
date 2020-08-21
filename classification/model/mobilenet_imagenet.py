import torchvision.models as models


def make_model(args, parent=False):
    # kwargs = {'width_mult': args.width_mult}
    return models.mobilenet_v2(pretrained=False, width_mult = args[0].width_mult)