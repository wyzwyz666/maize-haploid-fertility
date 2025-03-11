import os
from threading import local
import sys
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(backbone, model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0
    backbone     = backbone
    val_loss        = 0
    val_accuracy    = 0
    val_top3_accuracy = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=2.0)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc_train'  : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(5)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=2.0)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs     = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss    += loss_value.item()
            # # 计算Top-1准确率
            # accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            # val_accuracy    += accuracy.item()
            # 初始化正确预测数量和总预测数量
            correct_predictions = 0
            total_predictions = 0

            # 计算每个batch的正确预测数量和总预测数量
            correct_predictions += (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).sum().item()
            total_predictions += targets.size(0)

            # 计算Top-1准确率
            accuracy = correct_predictions / total_predictions
            val_accuracy += accuracy

            # 计算Top-3准确率
            _, top3 = torch.topk(F.softmax(outputs, dim=-1), 3, dim=-1)
            correct = top3.eq(targets.view(-1, 1).expand_as(top3))
            top3_accuracy = correct.any(dim=1).float().mean()
            val_top3_accuracy += top3_accuracy.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'acc_val'  : val_accuracy / (iteration + 1),
                                'top3_val'  : val_top3_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(5)



    if local_rank == 0:
        pbar.n = epoch_step_val
        pbar.refresh()
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, total_accuracy / epoch_step, val_accuracy / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        # print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        print('Acc_Val: %.4f ' % (val_accuracy / epoch_step_val))
        # print('Top3_Val: %.3f ' % (val_top3_accuracy / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir,f"ep{epoch + 1}-train_acc{total_accuracy / epoch_step:.3f}-val_acc{val_accuracy / epoch_step_val:.3f}-{backbone}.pth"))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            # torch.save(model.state_dict(), os.path.join(save_dir, f"best_epoch_weights-{backbone}.pth"))
            
        # torch.save(model.state_dict(), os.path.join(save_dir, f"last_epoch_weights-{backbone}.pth"))
