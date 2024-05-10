from copy import deepcopy
import numpy as np
import torch
import lightning as l
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_fabric.accelerators import find_usable_cuda_devices
from utils.config_parser import ConfigParserYaml
from graph.model import ModelBaseline
from data import get_dataloader
from utils import get_num_devices, setup_seed


if __name__ == "__main__":
    # load configuration
    parser = ConfigParserYaml(description='Training Configuration')
    args = parser.parse()
    # setup training
    setup_seed(seed=args.ExpConfig.seed, use_cuda=args.ExpConfig.use_cuda, cudnn_benchmark=args.ExpConfig.cudnn_benchmark)
    torch.set_float32_matmul_precision("high")
    num_devices = get_num_devices(use_cuda=args.ExpConfig.use_cuda) if isinstance(args.ExpConfig.num_devices, int) and args.ExpConfig.num_devices == -1 else args.ExpConfig.num_devices
    if (isinstance(num_devices, list) and len(num_devices) > 1) or (isinstance(num_devices, int) and num_devices > 1):
        strategy = "ddp_find_unused_parameters_false"
    else:
        strategy = "auto"
    # setup callbacks
    base_pl_callbacks = [
        LearningRateMonitor(logging_interval="epoch")
    ]
    if args.ExpConfig.summary:
        # setup summary callback
        from pytorch_lightning.callbacks import ModelSummary
        base_pl_callbacks += [ModelSummary()]
    # setup logger
    if args.ExpConfig.log == "tensorboard":
        # tensorboard logger
        from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=args.ExpConfig.exp_path, name="logs", max_queue=100)
    elif args.ExpConfig.log == "neptune":
        # neptune logger
        from pytorch_lightning.loggers.neptune import NeptuneLogger
        logger = NeptuneLogger(
            project=f"<NEPTUNE_PROJECT_NAME>",
            api_key="<NEPTUNE_API_KEY>",
            log_model_checkpoints=False,
            tags=args.ExpConfig.tags,
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            name=args.ExpConfig.name,
        )
    else:
        # csv logger
        from pytorch_lightning.loggers.csv_logs import CSVLogger
        logger = CSVLogger(save_dir=args.ExpConfig.exp_path, name='logs')
    logger.log_hyperparams(args)
    # setup model
    network_configs = deepcopy(args.NetworkConfig)
    criterion_configs = deepcopy(args.LossConfig)
    optimizer_configs = deepcopy(args.OptimConfig)
    scheduler_configs = deepcopy(args.SchedulerConfig)
    metric_configs = deepcopy(args.MetricConfig)
    model = ModelBaseline(network_kwargs=network_configs,
                          criterion_kwargs=criterion_configs,
                          optimizer_kwargs=optimizer_configs,
                          scheduler_kwargs=scheduler_configs,
                          metric_kwargs=metric_configs,
                          logger=args.ExpConfig.log)
    # set model summarization
    if args.ExpConfig.summary:
        if args.ExpConfig.log == 'neptune':
            logger.log_model_summary(model)
    # get dataloader
    data_configs = deepcopy(args.DataConfig)
    data = get_dataloader(mode=args.ExpConfig.split, **data_configs)

    best_ckpts, best_accs, best_baccs, best_mccs, best_f1s = [], [], [], [], []

    # setup trainer
    for k in range(1, args.ExpConfig.kfold + 1):
        callbacks = [ModelCheckpoint(dirpath=args.ExpConfig.exp_path + f"/checkpoints_fold{k}",
                                     filename="model-{epoch:02d}",
                                     monitor="val/accuracy",
                                     mode="max",
                                     save_last=False,
                                     every_n_epochs=1,
                                     save_weights_only=True,
                                     save_on_train_epoch_end=True)] + base_pl_callbacks
        data.setup(stage="fit", k=k)
        num_batches = len(data.train_dataloader()) // (len(num_devices) if isinstance(num_devices, list) else num_devices)
        model.reset_parameters()
        trainer = l.Trainer(
            accelerator="gpu" if args.ExpConfig.use_cuda else "cpu",
            devices=find_usable_cuda_devices(num_devices) if isinstance(num_devices, int) else num_devices,
            strategy=strategy,
            logger=logger,
            callbacks=callbacks,
            max_epochs=args.ExpConfig.num_epochs,
            num_sanity_val_steps=0,
            log_every_n_steps=num_batches,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0
        )
        # train model
        trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
        # test model
        data.setup(stage="test", k=k)
        trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path="best")
        # save best model
        best_accs.append(model.best_acc.cpu().item())
        best_baccs.append(model.best_bacc.cpu().item())
        best_mccs.append(model.best_mcc.cpu().item())
        best_f1s.append(model.best_f1.cpu().item())
        best_ckpts.append(callbacks[0].best_model_path)
    print(f"Test accuracy: {best_accs}")
    print(f"Mean accuracy: {np.array(best_accs).mean()}")
    print(f"Std accuracy: {np.array(best_accs).std()}")
    print(f"\nTest balanced accuracy: {best_baccs}")
    print(f"Mean balanced accuracy: {np.array(best_baccs).mean()}")
    print(f"Std balanced accuracy: {np.array(best_baccs).std()}")
    print(f"\nTest MCC: {best_mccs}")
    print(f"Mean MCC: {np.array(best_mccs).mean()}")
    print(f"Std MCC: {np.array(best_mccs).std()}")
    print(f"\nTest F1: {best_f1s}")
    print(f"Mean F1: {np.array(best_f1s).mean()}")
    print(f"Std F1: {np.array(best_f1s).std()}")
    # save best network checkpoint after training
    print("\nSaving best model...")
    model = ModelBaseline.load_from_checkpoint(best_ckpts[best_accs.index(max(best_accs))],
                                               network_kwargs=network_configs,
                                               criterion_kwargs=criterion_configs,
                                               optimizer_kwargs=optimizer_configs,
                                               scheduler_kwargs=scheduler_configs,
                                               metric_kwargs=metric_configs,
                                               logger=args.ExpConfig.log)
    torch.save(model.net.state_dict(), args.ExpConfig.exp_path + "/net_best.pt")
    print("Done!")
