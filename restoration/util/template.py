def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template == 'EDSR_CLUSTER':
        args.base = 'EDSR_CLUSTER'
        args.base_p = 'EDSR_CLUSTER'

    if args.template == 'DNCNN_CLUSTER':
        args.base = 'DNCNN_CLUSTER'
        args.base_p = 'DNCNN_CLUSTER'

    if args.template == 'SRRESNET_CLUSTER':
        args.base = 'SRRESNET_CLUSTER'
        args.base_p = 'SRRESNET_CLUSTER'

    if args.template == 'UNETDN5_CLUSTER':
        args.base = 'UNETDN5_CLUSTER'
        args.base_p = 'UNETDN5_CLUSTER'

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.lr_decay = 500
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.lr_decay = 150

