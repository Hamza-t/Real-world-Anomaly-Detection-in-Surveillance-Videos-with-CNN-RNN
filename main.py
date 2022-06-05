import numpy as np
from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
# from lstm import Lstm
import os
from sklearn import metrics
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Learner(input_dim=2048).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    #print('loss = ', train_loss / len(normal_train_loader))
    scheduler.step()
    return train_loss / len(normal_train_loader)


def test_abnormal(epoch):
    model.eval()
    matrix = np.zeros((2,2))
    auc = 0
    recall = 0
    precision = 0
    f_score = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cpu'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            # print(score)
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))

            for j in range(32):
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cpu'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            # print(len(gt_list3)) #binaire
            # print(len(score_list3)) #non binare
            score_final = [1 if i >= 0.5 else 0 for i in score_list3]
            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            matrix += metrics.confusion_matrix(y_true=gt_list3, y_pred=score_final)

            precision += metrics.precision_recall_fscore_support(gt_list3, score_final, pos_label=1)[0][0]
            recall += metrics.precision_recall_fscore_support(gt_list3, score_final, pos_label=1)[1][0]
            f_score += metrics.precision_recall_fscore_support(gt_list3, score_final, pos_label=1)[2][0]
            auc += metrics.auc(fpr, tpr)  # False negative !!
        print(fpr)
        print(tpr)
        print(matrix)
        auc = auc/140
        precision = precision / 140
        recall = recall / 140
        f_score = f_score / 140


        return auc, precision, recall, f_score, fpr, tpr


loss_vector = []
auc_vector = []
precision_vector = []
recall_vector = []
f_score_vector = []
for epoch in range(0,75):
    loss = train(epoch)
    auc, precision, recall, f_score, fpr, tpr = test_abnormal(epoch)

    '''
    loss_vector.append(loss)
    auc_vector.append(auc)
    precision_vector.append(precision)
    recall_vector.append(recall)
    f_score_vector.append(f_score)

data = {'loss': loss_vector,
        'auc': auc_vector,
        'precision': precision_vector,
        'recall': recall_vector,
        'F_score': f_score_vector
        }
df = pd.DataFrame(data)
df.to_csv("pfa.csv")
'''