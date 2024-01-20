import torch
import torch.nn as nn
import torch.nn.functional as F

class ReKT(nn.Module):
    def __init__(self, pro_max, skill_max, d, p):
        super(ReKT, self).__init__()

        self.pro_max = pro_max
        self.skill_max = skill_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))

        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.out = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )

        self.dropout = nn.Dropout(p=p)

        self.time_embed = nn.Parameter(torch.rand(200, d))

        self.ls_state = nn.Parameter(torch.rand(1, d))
        self.c_state = nn.Parameter(torch.rand(1, d))

        self.pro_state = nn.Parameter(torch.rand(199, d))
        self.skill_state = nn.Parameter(torch.rand(199, d))

        self.obtain_pro_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_pro_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )

        self.obtain_all_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

        self.obtain_skill_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_skill_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )
        self.obtain_all_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )

        self.akt_pro_diff = nn.Parameter(torch.rand(pro_max, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(skill_max, d))

    def forward(self, last_problem, last_skill, last_ans, next_problem, next_skill, next_ans):
        device = last_problem.device
        batch = last_problem.shape[0]
        seq = last_problem.shape[-1]

        next_pro_embed = F.embedding(next_problem, self.pro_embed) + F.embedding(next_skill,
                                                                                 self.skill_embed) + F.embedding(
            next_problem, self.akt_pro_diff) * F.embedding(next_skill, self.akt_pro_change)

        next_X = next_pro_embed + F.embedding(next_ans.long(), self.ans_embed)

        last_pro_time = torch.zeros((batch, self.pro_max)).to(device)  # batch pro
        last_skill_time = torch.zeros((batch, self.skill_max)).to(device)  # batch skill

        pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d
        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        all_state = self.ls_state.repeat(batch, 1)  # batch d

        last_pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d
        last_skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        batch_index = torch.arange(batch).to(device)

        all_time_gap = torch.ones((batch, seq)).to(device)
        all_time_gap_embed = F.embedding(all_time_gap.long(), self.time_embed)  # batch seq d

        res_p = []

        for now_step in range(seq):
            now_pro_embed = next_pro_embed[:, now_step]  # batch d

            now_item_pro = next_problem[:, now_step]  # batch
            now_item_skill = next_skill[:, now_step]

            last_batch_pro_time = last_pro_time[batch_index, now_item_pro]  # batch
            last_batch_pro_state = pro_state[batch_index, last_batch_pro_time.long()]  # batch d

            time_gap = now_step - last_batch_pro_time  # batch
            time_gap_embed = F.embedding(time_gap.long(), self.time_embed)  # batch d

            last_batch_skill_time = last_skill_time[batch_index, now_item_skill]  # batch
            last_batch_skill_state = skill_state[batch_index, last_batch_skill_time.long()]  # batch d

            skill_time_gap = now_step - last_batch_skill_time  # batch
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), self.time_embed)  # batch d

            item_pro_state_forget = self.obtain_pro_forget(
                self.dropout(torch.cat([last_batch_pro_state, time_gap_embed], dim=-1)))
            last_batch_pro_state = last_batch_pro_state * item_pro_state_forget

            item_skill_state_forget = self.obtain_skill_forget(
                self.dropout(torch.cat([last_batch_skill_state, skill_time_gap_embed], dim=-1)))
            last_batch_skill_state = last_batch_skill_state * item_skill_state_forget

            item_all_state_forget = self.obtain_all_forget(
                self.dropout(torch.cat([all_state, all_time_gap_embed[:, now_step]], dim=-1)))
            last_batch_all_state = all_state * item_all_state_forget

            last_pro_state[:, now_step] = last_batch_pro_state
            last_skill_state[:, now_step] = last_batch_skill_state

            final_state = torch.cat(
                [last_batch_all_state, last_batch_pro_state, last_batch_skill_state, now_pro_embed], dim=-1)

            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)

            res_p.append(P)

            item_all_obtain = self.obtain_all_state(
                self.dropout(torch.cat([last_batch_all_state, next_X[:, now_step]], dim=-1)))
            item_all_state = last_batch_all_state + torch.tanh(item_all_obtain)

            all_state = item_all_state

            pro_get = next_X[:, now_step]
            skill_get = next_X[:, now_step]

            item_pro_obtain = self.obtain_pro_state(
                self.dropout(torch.cat([last_batch_pro_state, pro_get], dim=-1)))
            item_pro_state = last_batch_pro_state + torch.tanh(item_pro_obtain)

            item_skill_obtain = self.obtain_skill_state(
                self.dropout(torch.cat([last_batch_skill_state, skill_get], dim=-1)))
            item_skill_state = last_batch_skill_state + torch.tanh(item_skill_obtain)

            last_pro_time[batch_index, now_item_pro] = now_step
            pro_state[:, now_step] = item_pro_state

            last_skill_time[batch_index, now_item_skill] = now_step
            skill_state[:, now_step] = item_skill_state

        res_p = torch.vstack(res_p).T

        return res_p

class ReKT_concept(nn.Module):
    def __init__(self, pro_max, skill_max, d, p):
        super(ReKT_concept, self).__init__()

        self.pro_max = pro_max
        self.skill_max = skill_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))

        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.out = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )

        self.dropout = nn.Dropout(p=p)

        self.time_embed = nn.Parameter(torch.rand(200, d))

        self.ls_state = nn.Parameter(torch.rand(1, d))
        self.c_state = nn.Parameter(torch.rand(1, d))

        self.pro_state = nn.Parameter(torch.rand(199, d))
        self.skill_state = nn.Parameter(torch.rand(199, d))

        self.obtain_pro_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_pro_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )

        self.obtain_all_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )

        self.obtain_skill_forget = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Sigmoid()
        )
        self.obtain_skill_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )
        self.obtain_all_state = nn.Sequential(
            nn.Linear(2 * d, d)
        )

        self.akt_pro_diff = nn.Parameter(torch.rand(pro_max, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(skill_max, d))

    def forward(self, last_problem, last_skill, last_ans, next_problem, next_skill, next_ans):

        device = last_skill.device
        batch = last_skill.shape[0]
        seq = last_skill.shape[-1]

        next_pro_embed = F.embedding(next_skill, self.skill_embed)

        next_X = next_pro_embed + F.embedding(next_ans.long(), self.ans_embed)

        last_skill_time = torch.zeros((batch, self.skill_max)).to(device)  # batch skill

        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        all_state = self.ls_state.repeat(batch, 1)  # batch d

        last_skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        batch_index = torch.arange(batch).to(device)

        all_time_gap = torch.ones((batch, seq)).to(device)
        all_time_gap_embed = F.embedding(all_time_gap.long(), self.time_embed)  # batch seq d

        res_p = []

        for now_step in range(seq):
            now_pro_embed = next_pro_embed[:, now_step]  # batch d

            now_item_skill = next_skill[:, now_step]

            last_batch_skill_time = last_skill_time[batch_index, now_item_skill]  # batch
            last_batch_skill_state = skill_state[batch_index, last_batch_skill_time.long()]  # batch d

            skill_time_gap = now_step - last_batch_skill_time  # batch
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), self.time_embed)  # batch d

            item_skill_state_forget = self.obtain_skill_forget(
                self.dropout(torch.cat([last_batch_skill_state, skill_time_gap_embed], dim=-1)))
            last_batch_skill_state = last_batch_skill_state * item_skill_state_forget

            item_all_state_forget = self.obtain_all_forget(
                self.dropout(torch.cat([all_state, all_time_gap_embed[:, now_step]], dim=-1)))
            last_batch_all_state = all_state * item_all_state_forget

            last_skill_state[:, now_step] = last_batch_skill_state

            final_state = torch.cat([last_batch_all_state, last_batch_skill_state, now_pro_embed], dim=-1)

            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)

            res_p.append(P)

            item_all_obtain = self.obtain_all_state(
                self.dropout(torch.cat([last_batch_all_state, next_X[:, now_step]], dim=-1)))
            item_all_state = last_batch_all_state + torch.tanh(item_all_obtain)

            all_state = item_all_state

            skill_get = next_X[:, now_step]

            item_skill_obtain = self.obtain_skill_state(
                self.dropout(torch.cat([last_batch_skill_state, skill_get], dim=-1)))
            item_skill_state = last_batch_skill_state + torch.tanh(item_skill_obtain)

            last_skill_time[batch_index, now_item_skill] = now_step
            skill_state[:, now_step] = item_skill_state

        res_p = torch.vstack(res_p).T

        return res_p