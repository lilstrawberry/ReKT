import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from model import ReKT, ReKT_concept
from run import run_epoch

mp2path = {
    'assist09': {
        'ques_skill_path': 'data/ASSIST09/ques_skill.csv',
        'train_path': 'data/ASSIST09/train_question.txt',
        'test_path': 'data/ASSIST09/test_question.txt',
        'train_skill_path': 'data/ASSIST09/train_skill.txt',
        'test_skill_path': 'data/ASSIST09/test_skill.txt',
        'skill_max': 167
    },
    'assist17': {
        'ques_skill_path': 'data/ASSIST17/ques_skill.csv',
        'train_path': 'data/ASSIST17/train_question.txt',
        'test_path': 'data/ASSIST17/test_question.txt',
        'train_skill_path': 'data/ASSIST17/train_skill.txt',
        'test_skill_path': 'data/ASSIST17/test_skill.txt',
        'skill_max': 86
    },
    'static11': {
        'ques_skill_path': 'data/Statics2011/ques_skill.csv',
        'train_path': 'data/Statics2011/train_question.txt',
        'test_path': 'data/Statics2011/test_question.txt',
        'train_skill_path': 'data/Statics2011/train_skill.txt',
        'test_skill_path': 'data/Statics2011/test_skill.txt',
        'skill_max': 106
    },
    'ednet': {
        'ques_skill_path': 'data/EdNet/ques_skill.csv',
        'train_path': 'data/EdNet/train_question.txt',
        'test_path': 'data/EdNet/test_question.txt',
        'train_skill_path': 'data/EdNet/train_skill.txt',
        'test_skill_path': 'data/EdNet/test_skill.txt',
        'skill_max': 1837
    },
    'eedi': {
        'ques_skill_path': 'data/Eedi/ques_skill.csv',
        'train_path': 'data/Eedi/train_question.txt',
        'test_path': 'data/Eedi/test_question.txt',
        'train_skill_path': 'data/Eedi/train_skill.txt',
        'test_skill_path': 'data/Eedi/test_skill.txt',
        'skill_max': 1050
    },
    'assist12': {
        'ques_skill_path': 'data/ASSIST12/ques_skill.csv',
        'train_path': 'data/ASSIST12/train_question.txt',
        'test_path': 'data/ASSIST12/test_question.txt',
        'train_skill_path': 'data/ASSIST12/train_skill.txt',
        'test_skill_path': 'data/ASSIST12/test_skill.txt',
        'skill_max': 265
    },
    'assist15': {
        #'ques_skill_path': 'data/ASSIST15/ques_skill.csv',
        'train_path': 'data/ASSIST15/assist2015_train1.csv',
        'valid_path': 'data/ASSIST15/assist2015_valid1.csv',
        'test_path': 'data/ASSIST15/assist2015_test1.csv',
        'train_skill_path': 'data/ASSIST15/assist2015_train1.csv',
        'valid_skill_path': 'data/ASSIST15/assist2015_valid1.csv',
        'test_skill_path': 'data/ASSIST15/assist2015_test1.csv',
        'skill_max': 101
    }
}

if __name__ == '__main__':

    for dataset in ['static11', 'assist17', 'assist09', 'eedi', 'ednet', 'assist15', 'assist12']:

        with open(f'{dataset}_output.txt', 'w') as file:
            # 遍历列表中的嵌套列表并写入文件
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if 'ques_skill_path' in mp2path[dataset]:
                ques_skill_path = mp2path[dataset]['ques_skill_path']

            train_path = mp2path[dataset]['train_path']

            if 'valid_path' in mp2path[dataset]:
                valid_path = mp2path[dataset]['valid_path']
            else:
                valid_path = mp2path[dataset]['test_path']

            test_path = mp2path[dataset]['test_path']

            train_skill_path = mp2path[dataset]['train_skill_path']

            if 'valid_skill_path' in mp2path[dataset]:
                valid_skill_path = mp2path[dataset]['valid_skill_path']
            else:
                valid_skill_path = mp2path[dataset]['test_skill_path']

            test_skill_path = mp2path[dataset]['test_skill_path']

            skill_max = mp2path[dataset]['skill_max']

            if 'ques_skill_path' in mp2path[dataset]:
                pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
            else:
                pro_max = skill_max

            p = 0.4

            d = 128
            learning_rate = 0.002
            epochs = 70
            batch_size = 80
            min_seq = 3
            max_seq = 200
            grad_clip = 15.0

            patience = 15

            avg_auc = 0
            avg_acc = 0

            sublist = []

            for now_step in range(5):

                best_acc = 0
                best_auc = 0
                state = {'auc': 0, 'acc': 0, 'loss': 0}

                model = ReKT(pro_max, skill_max, d, p)
                model = model.to(device)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

                one_p = 0

                for epoch in range(200):

                    one_p += 1

                    train_loss, train_acc, train_auc = run_epoch(pro_max, train_path, train_skill_path, batch_size,
                                                                 True, min_seq, max_seq, model, optimizer, criterion,
                                                                 device,
                                                                 grad_clip)
                    print(
                        f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

                    valid_loss, valid_acc, valid_auc = run_epoch(pro_max, valid_path, valid_skill_path, batch_size, False,
                                                              min_seq, max_seq, model, optimizer, criterion, device,
                                                              grad_clip)

                    print(
                        f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}')

                    sublist.append(valid_auc)

                    if valid_auc > best_auc:
                        one_p = 0
                        best_auc = valid_auc
                        best_acc = valid_acc
                        torch.save(model.state_dict(), f"./ReKT_{dataset}_{now_step}_model.pkl")
                        state['auc'] = valid_auc
                        state['acc'] = valid_acc
                        state['loss'] = valid_loss
                        torch.save(state, f'./ReKT_{dataset}_{now_step}_state.ckpt')

                    if one_p >= patience:
                        break

                model.load_state_dict(torch.load(f'./ReKT_{dataset}_{now_step}_model.pkl'))

                test_loss, test_acc, test_auc = run_epoch(pro_max, test_path, test_skill_path, batch_size, False,
                                                             min_seq, max_seq, model, optimizer, criterion, device,
                                                             grad_clip)

                print(f'*******************************************************************************')
                print(f'test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')
                print(f'*******************************************************************************')

                avg_auc += test_auc
                avg_acc += test_acc

            avg_auc = avg_auc / 5
            avg_acc = avg_acc / 5
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')
            print(f'*******************************************************************************')

            line = '\n'.join(str(item) for item in sublist)
            file.write(line)