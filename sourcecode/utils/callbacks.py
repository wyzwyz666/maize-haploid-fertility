import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, backbone, log_dir, model, input_shape):
        backbone = backbone
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        # self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.log_dir = os.path.join(log_dir, f"loss_{backbone}_{time_str}")
        self.losses     = []
        self.val_loss   = []
        self.total_accuracy = []
        self.val_accuracy = []
        self.val_accbest = 0.1

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss, total_accuracy, val_accuracy):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.total_accuracy.append(total_accuracy)
        self.val_accuracy.append(val_accuracy)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_train_acc.txt"), 'a') as f:
            f.write(str(total_accuracy))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(str(val_accuracy))
            f.write("\n")

        current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        with open(os.path.join(self.log_dir, "iteration_time_log.txt"), 'a') as f:
            f.write(current_time)
            f.write("\n")
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('train_acc', total_accuracy, epoch)
        self.writer.add_scalar('val_acc', val_accuracy, epoch)
        self.loss_plot()
        self.acc_plot()

    # def append_acc(self, epoch, train_acc, val_acc):
    #     self.train_acc.append(train_acc)
    #     self.val_acc.append(val_acc)
    #     self.val_accbest = max(self.val_accbest, val_acc)

    #     with open(os.path.join(self.log_dir, "epoch_train_acc.txt"), 'a') as f:
    #         f.write(str(train_acc))
    #         f.write("\n")
    #     with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
    #         f.write(str(val_acc))
    #         f.write("\n")


    #     self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
    #     self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='test loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth test loss')
            # plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'blue', linestyle = '--', linewidth = 2, label='smooth train acc')
            # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'purple', linestyle = '--', linewidth = 2, label='smooth test acc')

        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.total_accuracy, 'blue', linewidth=2, label='train accuracy')
        plt.plot(iters, self.val_accuracy, 'purple', linewidth=2, label='validation accuracy')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.total_accuracy, num, 3), 'blue', linestyle = '--', linewidth = 2, label='smooth train acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accuracy, num, 3), 'purple', linestyle = '--', linewidth = 2, label='smooth val acc')

        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")
