from copy import deepcopy
import numpy as np
import torch.distributed
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_fabric.accelerators import find_usable_cuda_devices
from utils.config_parser import ConfigParserYaml
from graph.model import ModelBaseline
from data import get_dataloader
from utils import setup_seed, get_num_devices


if __name__ == "__main__":
    # load configuration
    parser = ConfigParserYaml(description="Training Configuration")
    args = parser.parse()
    # setup training
    setup_seed(seed=args.ExpConfig.seed, use_cuda=args.ExpConfig.use_cuda, cudnn_benchmark=args.ExpConfig.cudnn_benchmark)
    torch.set_float32_matmul_precision("high")
    num_devices = get_num_devices(use_cuda=args.ExpConfig.use_cuda) if isinstance(args.ExpConfig.num_devices, int) and args.ExpConfig.num_devices == -1 else args.ExpConfig.num_devices
    if (isinstance(num_devices, list) and len(num_devices) > 1) or (isinstance(num_devices, int) and num_devices > 1):
        distributed = True
        strategy = "ddp_find_unused_parameters_false"
    else:
        distributed = False
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
        logger = CSVLogger(save_dir=args.ExpConfig.exp_path, name="logs")
    logger.log_hyperparams(args)
    # setup pretrain/transferring models
    pretrain_network_configs = deepcopy(args.NetworkConfig)
    pretrain_criterion_configs = deepcopy(args.LossConfig)
    pretrain_optimizer_configs = deepcopy(args.OptimConfig)
    pretrain_metric_configs = deepcopy(args.MetricConfig)
    pretrain_scheduler_configs = deepcopy(args.SchedulerConfig)
    model_pretrain = ModelBaseline(network_kwargs=pretrain_network_configs,
                                   criterion_kwargs=pretrain_criterion_configs,
                                   optimizer_kwargs=pretrain_optimizer_configs,
                                   scheduler_kwargs=pretrain_scheduler_configs,
                                   metric_kwargs=pretrain_metric_configs,
                                   logger=args.ExpConfig.log)
    transfer_network_configs = deepcopy(args.NetworkConfig)
    transfer_criterion_configs = deepcopy(args.LossConfig)
    transfer_optimizer_configs = deepcopy(args.OptimConfig)
    transfer_metric_configs = deepcopy(args.MetricConfig)
    transfer_scheduler_configs = deepcopy(args.SchedulerConfig)
    if transfer_scheduler_configs["name"] == "cosine_onecycle":
        transfer_scheduler_configs["T_max"] = 10
        transfer_scheduler_configs["T_start"] = 5
    model_transfer = ModelBaseline(network_kwargs=transfer_network_configs,
                                   criterion_kwargs=transfer_criterion_configs,
                                   optimizer_kwargs=transfer_optimizer_configs,
                                   scheduler_kwargs=transfer_scheduler_configs,
                                   metric_kwargs=transfer_metric_configs,
                                   logger=args.ExpConfig.log)
    if args.ExpConfig.summary:
        if args.ExpConfig.log == "neptune":
            logger.log_model_summary(model_pretrain)
    data_configs = deepcopy(args.DataConfig)
    data = get_dataloader(**data_configs)
    # setup trainer
    best_accs = {i: [] for i in range(1, args.ExpConfig.kfold + 1)}
    best_baccs = {i: [] for i in range(1, args.ExpConfig.kfold + 1)}
    best_mccs = {i: [] for i in range(1, args.ExpConfig.kfold + 1)}
    best_f1s = {i: [] for i in range(1, args.ExpConfig.kfold + 1)}
    best_ckpts = {i: [] for i in range(1, args.ExpConfig.kfold + 1)}
    #
    num_epochs_pretrain = args.ExpConfig.num_epochs
    num_epochs_transfer = 20
    #
    for k in range(1, args.ExpConfig.kfold + 1):
        # pretrain
        print("Training pretrain model...")
        callbacks_pretrain = [ModelCheckpoint(dirpath=args.ExpConfig.exp_path + f"/checkpoints_pretrain_fold{k}",
                                              filename="model-{epoch:02d}",
                                              monitor="val/accuracy",
                                              mode="max",
                                              save_last=False,
                                              every_n_epochs=1,
                                              save_weights_only=True,
                                              save_on_train_epoch_end=True)] + base_pl_callbacks
        # train pretrain model
        data.setup(stage="fit", k=k, mode="pretrain")
        num_batches = len(data.train_dataloader()) // (len(num_devices) if isinstance(num_devices, list) else num_devices)
        trainer = pl.Trainer(
            accelerator="gpu" if args.ExpConfig.use_cuda else "cpu",
            devices=find_usable_cuda_devices(num_devices) if isinstance(num_devices, int) else num_devices,
            strategy=strategy,
            logger=logger,
            callbacks=callbacks_pretrain,
            max_epochs=num_epochs_pretrain,
            num_sanity_val_steps=0,
            log_every_n_steps=num_batches,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0
        )
        model_pretrain.reset_parameters()
        trainer.fit(model_pretrain, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
        # test pretrain model
        data.setup(stage="test", k=k, mode="pretrain")
        trainer.test(model_pretrain, dataloaders=data.test_dataloader(), ckpt_path="best")
        torch.save(model_pretrain.net.state_dict(), args.ExpConfig.exp_path + f"/checkpoints_pretrain_fold{k}/pretrain_net_best.pt")
        print("Done!")
        # transfer
        print("Training transferring model...")
        callbacks_transfer = [ModelCheckpoint(dirpath=args.ExpConfig.exp_path + f"/checkpoints_transfer_fold{k}",
                                              filename="model-{epoch:02d}",
                                              monitor="val/accuracy",
                                              mode="max",
                                              every_n_epochs=1,
                                              save_weights_only=True,
                                              save_on_train_epoch_end=True)] + base_pl_callbacks
        # train transfer model
        for t in range(1, args.ExpConfig.kfold + 1):
            if k == t:
                continue
            data.setup(stage="fit", k=k, mode="transfer")
            num_batches = len(data.train_dataloader()) // (len(num_devices) if isinstance(num_devices, list) else num_devices)
            trainer = pl.Trainer(
                accelerator="gpu" if args.ExpConfig.use_cuda else "cpu",
                devices=find_usable_cuda_devices(num_devices) if isinstance(num_devices, int) else num_devices,
                strategy=strategy,
                logger=logger,
                callbacks=callbacks_transfer,
                max_epochs=num_epochs_transfer,
                num_sanity_val_steps=0,
                log_every_n_steps=num_batches,
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0
            )
            model_transfer.reset_parameters()
            model_transfer.load_net_state_dict(args.ExpConfig.exp_path + f"/checkpoints_pretrain_fold{k}/pretrain_net_best.pt")
            model_transfer.freeze(proj=True, att=True, head=False)
            trainer.fit(model_transfer, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
            data.setup(stage="test", k=k, mode="transfer")
            trainer.test(model_transfer, dataloaders=data.test_dataloader())
            #
            best_accs[t].append(model_transfer.best_acc.detach().cpu().item())
            best_baccs[t].append(model_transfer.best_bacc.detach().cpu().item())
            best_mccs[t].append(model_transfer.best_mcc.detach().cpu().item())
            best_f1s[t].append(model_transfer.best_f1.detach().cpu().item())
            best_ckpts[t].append(callbacks_transfer[0].best_model_path)
    #
    for i in range(1, args.ExpConfig.kfold + 1):
        print(f"\nfold{i}")
        print(f"Mean transfer accuracy: {np.array(best_accs[i]).mean()}")
        print(f"Std transfer accuracy: {np.array(best_accs[i]).std()}")
        print(f"Mean transfer balanced accuracy: {np.array(best_baccs[i]).mean()}")
        print(f"Std transfer balanced accuracy: {np.array(best_baccs[i]).std()}")
        print(f"Mean transfer MCC: {np.array(best_mccs[i]).mean()}")
        print(f"Std transfer MCC: {np.array(best_mccs[i]).std()}")
        print(f"Mean transfer F1: {np.array(best_f1s[i]).mean()}")
        print(f"Std transfer F1: {np.array(best_f1s[i]).std()}")
