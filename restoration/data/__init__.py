from importlib import import_module
# from dataloader import MSDataLoader
from torch.utils.data.dataloader import DataLoader
# from IPython import embed
class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            # embed()
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(args, train=False)

        elif args.data_test in ['DenoiseSet68','DenoiseDIV2K20','DenoiseColorSet68', 'DenoiseDIV2K100Color', 'DenoiseDIV2K100Gray']:
            module_test = import_module('data.benchmarkdenoise')
            testset = getattr(module_test, 'BenchmarkDenoise')(args, train=False)

        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )