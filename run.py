from sklearn import metrics
from tqdm import tqdm
import torch
import numpy as np

from load_data import getLoader

def run_epoch(max_problem, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num,
              model, optimizer, criterion, device, grad_clip):
    total_correct = 0
    total_num = 0
    total_loss = []

    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(max_problem, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num)

    for i, data in tqdm(enumerate(data_loader), desc='加载中...'):

        last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, mask = data

        if is_train:
            predict = model(last_problem, last_skill, last_ans, next_problem, next_skill, next_ans)

            next_predict = torch.masked_select(predict, mask)
            next_true = torch.masked_select(next_ans, mask)

            loss = criterion(next_predict, next_true)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())

            total_loss.append(loss.item())
            total_num += len(next_true)

            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()
        else:
            with torch.no_grad():
                predict = model(last_problem, last_skill, last_ans, next_problem, next_skill, next_ans)

                next_predict = torch.masked_select(predict, mask)
                next_true = torch.masked_select(next_ans, mask)

                loss = criterion(next_predict, next_true)

                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())

                total_loss.append(loss.item())
                total_num += len(next_true)

                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

    avg_loss = np.average(total_loss)
    acc = total_correct * 1.0 / total_num
    auc = metrics.roc_auc_score(labels, outputs)
    return avg_loss, acc, auc