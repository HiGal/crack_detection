import torch


class TrainConfig:
    n_epochs = 10  # number of epochs
    batch_size = 64
    image_size = (64, 64)  # set an image size after resizing
    train_folder = "data/train"  # train folder
    val_folder = "data/test"  # validation folder
    folder = "efficientnet-b0"  # folder for trained model
    device = "cuda:0"
    num_workers = 8  # num_workers in dataloader
    lr = 1e-4        # learning rate
    verbose = True   # should information be printed
    verbose_step = 10 # number of iterations to print information about learning process
    step_scheduler = False          # schedulers
    validation_scheduler = True     # schedulers
    # Scheduler and its parameters
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
