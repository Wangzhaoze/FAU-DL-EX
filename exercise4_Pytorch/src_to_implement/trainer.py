import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self, model, crit, optim=None, train_dl=None, val_test_dl=None, cuda=True, early_stopping_patience=-1):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        checkpoint = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(checkpoint['state_dict'])

    def save_onnx(self, file):
        """
        :param file: file or file-like object used to save onnx
        :return: None
        """
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)

        t.onnx.export(m, x, file, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'},
                                                                                    'output': {0: 'batch_size'}})

    def train_step(self, input, y):
        # train by GPU
        if self._cuda:
            input = input.cuda()
            y = y.cuda()

        # reset parameters after one time
        self._model.zero_grad()

        # calculate through forward network
        output = self._model.forward(input)

        # define loss
        loss = self._crit(output, y.detach())

        # Backpropagation
        loss.backward()

        # optimizer
        self._optim.step()
        return loss

    def train_epoch(self):
        # define loss and start train model
        loss = 0
        self._model.train()

        # train by GPU
        if self._cuda:
            self._model = self._model.cuda()
            self._crit = self._crit.cuda()

        # calculate loss with train dataset
        for img, label in tqdm(self._train_dl, desc='train'):
            img = img.requires_grad_(True)
            label = label.float().requires_grad_(True)
            loss += Trainer.train_step(self, img, label).item()

        # calculate average loss
        loss = loss / len(self._train_dl)
        return loss

    def val_test_step(self, input, y):

        # test by GPU
        if self._cuda:
            input = input.cuda()
            y = y.cuda()

        # calculate through forward network
        y_pred = self._model.forward(input).round()

        # calculate test loss
        loss = self._crit(y_pred, y.float())
        return loss, y_pred

    def val_test(self):
        # start test eval model
        self._model.eval()

        # define loss pre and labels
        loss_test = 0
        predicts = []
        labels = []

        # calculate loss with test dataset
        for img, label in tqdm(self._val_test_dl, desc='val'):
            img = img.requires_grad_(False)
            label = label.requires_grad_(False)
            if self._cuda:
                img = img.cuda()
                label = label.cuda()
            labels.append(label.cpu().tolist())
            loss, y_pred = Trainer.val_test_step(self, img, label)
            loss_test += loss.item()
            predicts.append(y_pred.cpu().tolist())

        # calculate average loss
        loss_test = loss_test / len(self._val_test_dl)
        return loss_test, predicts, labels

    def fit(self, epochs=-1):
        train_loss = []
        val_loss = []
        f1_list = []
        epoch = 0
        f1 = 0
        valmin = float('inf')
        finish_patience = 0
        while True:
            print('epoch ' + str(epoch))

            # calculate train and val_test loss
            t_loss = Trainer.train_epoch(self)
            v_loss, pred, labels = Trainer.val_test(self)

            # get labels and predictions
            labels = [item for sublist in labels for item in sublist]
            pred = [item for sublist in pred for item in sublist]

            # calculate F1-score
            f1score = f1_score(np.around(np.array(labels).flatten()),
                               np.around(np.array(pred).flatten()))
            f1_list.append(f1score)
            if v_loss < valmin or f1score > f1:
                valmin = v_loss
                finish_patience = 0
            else:
                finish_patience += 1

            # F1-score > 0.6, save model parameters
            if (f1score > f1) and (f1score > 0.6):
                Trainer.save_onnx(self, 'checkpoint_test.onnx')
                f1 = f1score
                Trainer.save_checkpoint(self, epoch + 1)
            print('f1 score: ' + str(f1score))
            print('training loss: ' + str(t_loss))
            print('validation loss: ' + str(v_loss))

            # finish one epoch, save loss results of train and test
            epoch += 1
            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # finish training conditions
            if epoch == epochs or finish_patience >= self._early_stopping_patience:
                print('f1_max: ' + str(f1))
                return train_loss, val_loss, f1_list
