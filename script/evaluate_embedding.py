# semiSupervised/evaluate_embedding.py
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

import warnings


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super().__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)


def logistic_classify(x, y, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        # test split
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs = torch.from_numpy(train_embs).to(device)
        train_lbls = torch.from_numpy(train_lbls).to(device)
        test_embs = torch.from_numpy(test_embs).to(device)
        test_lbls = torch.from_numpy(test_lbls).to(device)

        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad(set_to_none=True)
            loss = xent(log(train_embs), train_lbls)
            loss.backward()
            opt.step()

        log.eval()
        with torch.no_grad():
            preds = torch.argmax(log(test_embs), dim=1)
            acc = torch.mean((preds == test_lbls).float()).item()
        accs.append(acc)

        # val split (从 train_index 里再划一份)
        val_size = len(test_index)
        val_index = np.random.choice(train_index, val_size, replace=False).tolist()
        new_train_index = [i for i in train_index if i not in val_index]

        train_embs, val_embs = x[new_train_index], x[val_index]
        train_lbls, val_lbls = y[new_train_index], y[val_index]

        train_embs = torch.from_numpy(train_embs).to(device)
        train_lbls = torch.from_numpy(train_lbls).to(device)
        val_embs = torch.from_numpy(val_embs).to(device)
        val_lbls = torch.from_numpy(val_lbls).to(device)

        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad(set_to_none=True)
            loss = xent(log(train_embs), train_lbls)
            loss.backward()
            opt.step()

        log.eval()
        with torch.no_grad():
            preds = torch.argmax(log(val_embs), dim=1)
            acc = torch.mean((preds == val_lbls).float()).item()
        accs_val.append(acc)

    return float(np.mean(accs_val)), float(np.mean(accs))


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        val_index = np.random.choice(train_index, val_size, replace=False).tolist()
        new_train_index = [i for i in train_index if i not in val_index]

        x_train, x_val = x[new_train_index], x[val_index]
        y_train, y_val = y[new_train_index], y[val_index]

        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_val, classifier.predict(x_val)))

    return float(np.mean(accuracies_val)), float(np.mean(accuracies))


def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0.0
    acc_val = 0.0

    _acc_val, _acc = svc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    print(acc_val, acc)
    return acc_val, acc
