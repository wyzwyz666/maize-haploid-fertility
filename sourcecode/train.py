import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler, set_optimizer_lr, show_config, weights_init)
from utils.utils_fit import fit_one_epoch
from train_test_split import split_dataset_into_test_and_train_sets
CUDA_DEVICES = 0, 1, 2, 3, 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def objective(trial):
    # lr = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = True
    classes_path = 'model_data/cls_classes1.txt'
    backbone = "Maize-IRNet"
    input_shape = [299, 299]
    pretrained = False
    # model_path = "model_data/resnet101-5d3b4d8f.pth"
    model_path = ""
    Init_Epoch = 0
    Freeze_Epoch = 0
    batch_size = 16
    Freeze_batch_size = batch_size
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = batch_size
    Freeze_Train = False
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0  # adam
    # weight_decay = 5e-4  # sgd
    # lr_decay_type = "step"
    lr_decay_type = "cos"
    save_period = 1
    save_dir = 'logs/Maize-IRNet'
    num_workers = 12
    # Retrieve the paths of all files and their category indexes
    all_data_dir = 'datasets1/traintest'
    train_files, test_files, train_labels, test_labels = split_dataset_into_test_and_train_sets(all_data_dir,test_size=0.1)
    # Write the file paths and category indexes of the training and testing sets to a new txt file
    with open('clstrain.txt', 'w') as f:
        for file, label in zip(train_files, train_labels):
            f.write(str(label) + ';' + file + '\n')

    with open('clstest.txt', 'w') as f:
        for file, label in zip(test_files, test_labels):
            f.write(str(label) + ';' + file + '\n')

    train_annotation_path = './clstrain.txt'
    test_annotation_path = './clstest.txt'
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
    #  Download pre trained weights
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)
    #   get classes
    class_names, num_classes = get_classes(classes_path)
    if backbone not in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        #   Load based on the pre trained weight key and the model key
        model_dict = model.state_dict()
        # pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = torch.load(model_path)
        # pretrained_dict = torch.load(model_path, map_location='cuda:0')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #   Display no matching keys
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    #   Record Loss
    if local_rank == 0:
        loss_history = LossHistory(backbone, save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # Torch 1.2 does not support amp, it is recommended to use Torch 1.7.1 and above to correctly use FP16
    # Therefore, torch1.2 displays' could not be resolved 'here
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    
    #   Synchronize BN
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    if Cuda:
        if distributed:
            #   DataParallel
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],find_unused_parameters=True)
        else:
            # model_train = torch.nn.DataParallel(model)
            # cudnn.benchmark = True
            # model_train = model_train.cuda()
            model_train = model.to('cuda')
    #   Read the txt corresponding to the dataset
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            num_train=num_train, num_val=num_val
        )

    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
            optimizer_type, wanted_step))
        print(
            "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 128
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-7 if optimizer_type == 'adam' else 5e-4
        if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
            nbs = 256
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999),weight_decay=weight_decay),
            'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        train_dataset = DataGenerator(train_lines, input_shape, True)
        val_dataset = DataGenerator(val_lines, input_shape, False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=True,
                         drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=True,
                             drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
        
        #   Train
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 128
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-7 if optimizer_type == 'adam' else 5e-4
                if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                    nbs = 256
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                model.Unfreeze_backbone()
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=True,
                                 drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,pin_memory=True,
                                     drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(backbone, model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
        if local_rank == 0:
            loss_history.writer.close()
        return loss_history.val_accbest

if __name__ == "__main__":
    storage_name = "sqlite:///optunacihua.db"
    studyname = "cihua"
    study = optuna.create_study(storage=storage_name, pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",study_name=studyname,load_if_exists=True)
    study.optimize(objective, n_trials=1)
    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = " + str(best_value))
    print("best_params:")
    print(best_params)
