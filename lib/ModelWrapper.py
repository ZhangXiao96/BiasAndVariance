import torch
import torch.nn as nn
import numpy as np


class ModelWrapper(object):

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self._softmax = nn.Softmax(dim=-1)

    @staticmethod
    def activate_dropout(m):
        if (type(m) == nn.Dropout) or (type(m) == nn.Dropout2d):
            m.train()

    @staticmethod
    def inactivate_dropout(m):
        if (type(m) == nn.Dropout) or (type(m) == nn.Dropout2d):
            m.eval()

    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if type(self.criterion) == nn.MSELoss:
            outputs = self._softmax(outputs)
            one_hot_targets = torch.zeros(inputs.size(0), outputs.size(-1), device=self.device).scatter(1, targets.unsqueeze(-1), 1)
            loss = self.criterion(outputs, one_hot_targets)
        else:
            loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return loss.item(), acc, correct

    def eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        if type(self.criterion) == nn.MSELoss:
            outputs = self._softmax(outputs)
            one_hot_targets = torch.zeros(inputs.size(0), outputs.size(-1), device=self.device).scatter(1, targets.unsqueeze(-1), 1)
            loss = self.criterion(outputs, one_hot_targets)
        else:
            loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def predict_on_batch(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        outputs = self._softmax(outputs)
        _, predicted = outputs.max(1)
        return outputs.data.cpu().numpy(), predicted.data.cpu().numpy()

    def eval_all(self, test_loader):
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct = self.eval_on_batch(inputs, targets)
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx+1)
            test_acc = test_correct / total
        return test_loss, test_acc

    def predict_all(self, test_loader, max_number=None):
        with torch.no_grad():
            probs = []
            labels = []
            truth = []
            nb = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                prob, label = self.predict_on_batch(inputs)
                probs.append(prob)
                labels.append(label)
                truth.append(targets.cpu().numpy())
                nb += len(label)
                if max_number:
                    if nb >= max_number:
                        break
            nb_all = min(max_number, nb) if max_number else nb
            probs = np.concatenate(probs, axis=0)[:nb_all]
            labels = np.concatenate(labels, axis=0)[:nb_all]
            truth = np.concatenate(truth, axis=0)[:nb_all]
        return probs, labels, truth

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()
