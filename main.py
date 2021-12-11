import torch.nn as nn
import torch
import numpy as np
from zipfile import ZipFile
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import Counter
from sklearn.utils import shuffle
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd

from datetime import datetime
from pathlib import Path

import pytorch_util as ptu


assert len(sys.argv) > 1 and (sys.argv[1] == 'easy' or sys.argv[1] == 'hard'), '1st argument has to be \'easy\' or \'hard\''

data_path = 'data/' + sys.argv[1] + '/'
ptu.init_gpu()
ptu.set_device(0)

# Grid setup
easy_grid = sys.argv[1] == 'easy'

if easy_grid:
    positive_path = 'positive_data/grid_easy'
    characters = list(map(chr, range(65, 90)))
    grid_locations = np.array(characters)
    grid_shape = (5,5)
    characters = np.reshape(characters,grid_shape)
    walls = ['K', 'O', 'U','Y']
    objs = ['N']
    moves = {'left':0, 'right':1, 'up':2, 'down':3, 'pick':4}
    actions = list(moves.keys())
else:
    positive_path = 'positive_data/grid_hard'
    characters = [[letter+num for num in '123456789'] for letter in 'ABCDE']
    grid_locations = np.array(characters).flatten()
    grid_shape = (5,9)
    characters = np.reshape(characters,grid_shape)
    walls = ['A9', 'B3', 'C1', 'C5', 'C8', 'E1', 'E5']
    objs = ['A5', 'C7']
    moves = {'left':0, 'right':1, 'up':2, 'down':3}
    actions = list(moves.keys())


horizon = 15
pos_to_idx = {}
idx_to_pos = {}
print(characters)
h = len(characters) - 1

for (i, j), z in np.ndenumerate(characters):
    pos_to_idx[z] = (j,h-i)
    idx_to_pos[(j,h-i)] = (z)

if easy_grid:
    grid_keys = [k+k for k in list(pos_to_idx.keys())]
else:
    grid_keys = list(pos_to_idx.keys())
#not_valid = ['R']





def build_shortest_path_table(characters, walls, objs):
    def lee_shortest_path(start):
        dest_j, dest_i = pos_to_idx[start]

        table = np.full_like(characters, 10000, dtype=np.int)
        table[h-dest_i][dest_j] = 0
        queue = [(dest_j, dest_i)]
        while len(queue) != 0:
            j, i = queue.pop()
            l = (j-1, i)
            r = (j+1, i)
            u = (j, i+1)
            d = (j, i-1)
            cur_dist = table[h-i][j]
            if l in idx_to_pos and idx_to_pos[l] not in walls:
                if table[h-l[1]][l[0]] > cur_dist + 1:
                    table[h-l[1]][l[0]] = cur_dist + 1
                    queue.append(l)
            if r in idx_to_pos and idx_to_pos[r] not in walls:
                if table[h-r[1]][r[0]] > cur_dist + 1:
                    table[h-r[1]][r[0]] = cur_dist + 1
                    queue.append(r)
            if u in idx_to_pos and idx_to_pos[u] not in walls:
                if table[h-u[1]][u[0]] > cur_dist + 1:
                    table[h-u[1]][u[0]] = cur_dist + 1
                    queue.append(u)
            if d in idx_to_pos and idx_to_pos[d] not in walls:
                if table[h-d[1]][d[0]] > cur_dist + 1:
                    table[h-d[1]][d[0]] = cur_dist + 1
                    queue.append(d)
        return table

    def dist(location1, location2):
        loc1 = pos_to_idx[location1]
        loc2 = pos_to_idx[location2]
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    tables = []
    for obj in objs:
        tables.append(lee_shortest_path(obj))

    distance = 0
    for i in range(len(objs)-1, 0, -1):
       distance += dist(objs[i], objs[i-1])
       tables[i-1] += distance

    return tables

shortest_path_table = build_shortest_path_table(characters, walls, objs)
# print(shortest_path_table)


def expert_find_fault(trace, horizon):
    assert len(trace) <= horizon

    bad_pairs = set()
    last_state, last_action = trace[-1]

    # check if outside border
    outside_border = False
    if last_state in characters[0,:] and last_action is 'up': # ABCDE or A1~A9
        bad_pairs.add(trace[-1])
        outside_border = True
    if last_state in characters[-1,:] and last_action is 'down': # UVWXY or E1~E9
        bad_pairs.add(trace[-1])
        outside_border = True
    if last_state in characters[:,0] and last_action is 'left': # AFKPU or A1~E1
        bad_pairs.add(trace[-1])
        outside_border = True
    if last_state in characters[:,-1] and last_action is 'right': # EJOTY or A9~E9
        bad_pairs.add(trace[-1])
        outside_border = True

    # if len(trace) == 1: return list(bad_pairs)
    
    if not outside_border:
        x, y = pos_to_idx[last_state]
        end_state = None
        if   last_action == 'up': end_state = idx_to_pos[(x, y+1)]
        elif last_action == 'down': end_state = idx_to_pos[(x, y-1)]
        elif last_action == 'left': end_state = idx_to_pos[(x-1, y)]
        elif last_action == 'right': end_state = idx_to_pos[(x+1, y)]
        elif last_action == 'pick': end_state = idx_to_pos[(x, y)]
        # we don't need this because walls have distance oo, which we take care of below
        # if end_state in walls: bad_pairs.add(trace[-1])
        trace.append((end_state, ''))

    obj_idx = 0
    for step, ((s, a), (ns, na)) in enumerate(zip(trace[:-1], trace[1:])):
        # switch to next table if current goal is reached
        if s == objs[obj_idx] and not easy_grid:
            obj_idx += 1

        j, i = pos_to_idx[s]
        i = h - i
        nj, ni = pos_to_idx[ns]
        ni = h - ni

        next_step_left = horizon - step - 1
        if easy_grid: next_step_left -= 1 # last step should be 'pick'
        if shortest_path_table[obj_idx][ni][nj] > next_step_left:
            if shortest_path_table[obj_idx][i][j] <= shortest_path_table[obj_idx][ni][nj]:
                bad_pairs.add((s,a))
            
    if not outside_border: trace.pop()

    return list(bad_pairs)

# trace = [('B3',''), ('B4', ''), ('B5', ''), ('B6',''), ('A6',''), ('A5', ''), ('A6', 'up'), ('A7',''), ('A6', 'right')]
# trace = [('A',''), ('B',''), ('G',''), ('F',''), ('G',''), ('L',''), ('Q',''), ('V',''), ('W','up')]
# print(expert_find_fault(trace, 9))
# assert False


total_files = 0
no_counter_files = 0
demonstrations = []
demonstrations_name = []
# actions = ['left', 'right', 'up','down','pick']
object_bool = ['TRUE', "FALSE"]
object_pos = 'N'
flag_NR = False
for root,dirs,files in os.walk(positive_path):
    # print(root)
    for f in files:
        # print(f)
        # grid_num = int(root.split('_')[1])
        demo_len = f.split('.')[0].split('_')[-1]
        demo=[]
        total_files += 1 
        data = open(root+'/'+f).read().splitlines()
        flag_NR = False
        for idx,l in enumerate(data):
            if idx == 0 :
                continue
            else:
                tokens = l.split('\t')
                for t in tokens:
                    if t == 'NR':
                        flag_NR = True
                    if t in grid_keys:
                        state = t
                    if t in actions:
                        action = t
                    if t in object_bool:
                        obj = t
                if easy_grid:
                    demo.append((state,action,obj))
                else:
                    demo.append((state, action))
        
        if not(flag_NR):
            #if grid_num not in demonstrations:
            demonstrations.append([demo])
            demonstrations_name.append(f)




def get_positive_example(demonstrations, batch_size=20):
    states = []
    actions = []
    for traces in demonstrations:        
        for demo in traces:
            visit_a5 = False
            visit_c7 = False
            for d in demo:
                grid = np.zeros((3, *grid_shape))
                #getting location of agent and placing agent in grid
                state = d[0][0] if easy_grid else d[0]
                (x,y) = pos_to_idx[state]
                grid[0][h-y][x] = 1
                #placing walls in grid
                for w in walls:
                    (x,y) = pos_to_idx[w]
                    grid[1][h-y][x] = 1
                
                #placing object in grid
                if easy_grid:
                    if d[2] == 'TRUE':
                        (x,y) = pos_to_idx[object_pos] 
                        grid[2][h-y][x] = 1
                # changing state after visiting
                else:
                    (x_a5,y_a5) = pos_to_idx["A5"]
                    grid[2][5-y_a5][x_a5] = 0 if visit_a5 else 1

                    (x_c7,y_c7) = pos_to_idx["C7"]
                    grid[2][5-y_c7][x_c7] = 0 if visit_c7 else 1
                    
                actions.append(moves[d[1]])
                states.append(grid.flatten())
        
    assert len(states) == len(actions)
    states = np.array(states)
    actions = np.array(actions)
    states, actions = shuffle(states, actions)
    grid_dataset = GridDataset(states, actions)
    return DataLoader(grid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)




class GridDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, states, actions):
            self.states = states if isinstance(states, torch.Tensor) else ptu.from_numpy(states)
            self.actions = actions if isinstance(actions, torch.Tensor) else ptu.from_numpy(actions)
        def __len__(self):
            return len(self.states)
        def __getitem__(self, idx):
            #print(idx, states)
            data = (self.states[idx],self.actions[idx])
            return data



#def tensor_to_grid(grid):
#    grid = torch.reshape(grid,(3,5,5))
#    agent_grid = grid[0]
#    walls_grid = grid[1]
#    object_grid = grid[2]
#    walls = []
#    object_pos = ""
#    for i in range(len(agent_grid)):
#        for j in range(len(agent_grid[0])):
#            if agent_grid[i][j] == 1:
#                agent_pos = idx_to_pos[j,h-i]
#    for i in range(len(walls_grid)):
#        for j in range(len(walls_grid[0])):
#            if walls_grid[i][j] == 1:
#                walls.append(idx_to_pos[j,h-i])
#    for i in range(len(object_grid)):
#        for j in range(len(object_grid[0])):
#            if object_grid[i][j] == 1:
#                #print(i,j)
#                object_pos = idx_to_pos[j,h-i]
    
#    return agent_pos, walls, object_pos



class Network(nn.Module):
    def __init__(self): 
        super().__init__()
        num_hidden = 128 if easy_grid else 256
        input_dim = 3 * grid_shape[0] * grid_shape[1]
        output_dim = len(actions)
        self.fc1 = nn.Linear(input_dim,num_hidden)
        self.fc2 = nn.Linear(num_hidden,num_hidden)
        self.fc3 = nn.Linear(num_hidden,num_hidden)
        self.fc4 = nn.Linear(num_hidden,output_dim)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        #self.drop2 = nn.Dropout(0.1)
        self.softplus = nn.Softplus()
    def forward(self,state):
        state = self.drop1(self.relu(self.fc1(state)))
        state = self.relu(self.fc2(state))
        state = self.relu(self.fc3(state))
        state = self.softplus(self.fc4(state))
        return state




def generate_starting(loc, visit_a5=False):
    grid = torch.zeros((3,*grid_shape))
    (x,y) = pos_to_idx[loc]
    grid[0][h-y][x] = 1         
    for w in walls:
        (x,y) = pos_to_idx[w]
        grid[1][h-y][x] = 1

    if easy_grid:
        (x,y) = pos_to_idx[object_pos]
        grid[2][h-y][x] = 1
    else:
        (x_a5,y_a5) = pos_to_idx['A5']
        (x_c7,y_c7) = pos_to_idx['C7']
        grid[2][h-y_a5][x_a5] = 0 if visit_a5 else 1
        grid[2][h-y_c7][x_c7] = 1
    grid = grid.flatten()
    return grid.to(ptu.device)



def evaluate(model, num_trials, eval_method='argmax'):
    reward = 0
    for iter in range(num_trials):
        start = random.choice(list(set(grid_locations) - set(walls)))
        start_pos = start
        if easy_grid:
            pick = False
        else:
            visit_a5 = False
            visit_c7 = False
        nr = False
        for i in range(horizon):

            grid_start = generate_starting(start, visit_a5)
            model.eval()
            predictions = model(grid_start.float())
            probs = Categorical(F.softmax(predictions, dim=-1))
            if eval_method == 'sample':
                pred_action = probs.sample((1,))
            elif eval_method == 'argmax':
                pred_action = torch.argmax(F.softmax(predictions, dim=-1))[None]
            if pred_action[0] == 0:
                action_taken = 'left'
            elif pred_action[0] == 1:
                action_taken = 'right'
            elif pred_action[0] == 2:
                action_taken = 'up'
            elif pred_action[0] == 3:
                action_taken = 'down'
            else:
                action_taken = 'pick'

            start_loc = pos_to_idx[start]
        
            if action_taken == 'left':
                if (start_loc[0]-1,start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0]-1,start_loc[1])]
                else:
                    nr = True
            elif action_taken == 'right':
                if (start_loc[0]+1,start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0]+1,start_loc[1])]
                else:
                    nr = True
            elif action_taken == 'up':
                if (start_loc[0],start_loc[1]+1) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1]+1)]
                else:
                    nr = True
            elif action_taken == 'down':
                if (start_loc[0],start_loc[1]-1) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1]-1)]
                else:
                    nr = True
            elif action_taken == 'pick':
                if (start_loc[0],start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1])]
                else:
                    nr = True
            if nr or next_loc in walls:
                break

            if easy_grid:
                if next_loc == object_pos and action_taken == 'pick':
                    pick = True
                    break
            else:
                if next_loc == 'A5':
                    visit_a5 = True
                if next_loc == 'C7' and visit_a5:
                    visit_c7 = True
                    break

            start = next_loc
        
        if easy_grid:
            if pick and not nr:
                reward +=1
        else:
            if visit_a5 and visit_c7:
                reward += 1
        
        # else:
        #     print("start pos was ", start_pos,"pick ", pick,"Nr ",nr)
    # print("total average reward ", reward/10)
    return reward / num_trials



def train_with_positive_examples(model, dataloader, n_epochs, eval_method='argmax'):
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.001)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.01, last_epoch=-1, verbose=True)
    frequency = 10
    rewards_epoch = []
    iterations = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        total_accuracy = 0
        for data,labels in dataloader:
            data = data.float()
            labels = labels.long()
            assert model.training
            model.zero_grad()
            logits = model(data)
            loss = criterion(logits,labels)
            predictions = torch.argmax(F.softmax(logits, dim=-1),1)
            loss.backward()
            optimizer.step()
            batch_accuracy = (labels == predictions).float().sum()/len(labels)
            total_accuracy += batch_accuracy
            total_loss += loss
            num_batches += 1
        #scheduler.step()
        # print("EPOCH ", epoch, "Train Loss ", total_loss/num_batches, "Accuracy ", total_accuracy/num_batches)
        if epoch%frequency== 0:
            avg_reward = evaluate(model, 100, eval_method)
            rewards_epoch.append(avg_reward)
            iterations.append(epoch)
            assert len(iterations) == len(rewards_epoch)
    return iterations, rewards_epoch




def plot(x, y, xlabel='', ylabel='', title='', filename=''):
    assert len(x) == len(y), f'Mismatching length: x has length {len(x)} and y has length {len(y)}'
    plt.xticks(np.arange(0, max(x)+1, x[1]-x[0]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    if filename != '':
        plt.savefig(filename + '.png')
        pd.DataFrame({xlabel: x, ylabel: y}).to_csv(filename + '.csv', index=False)
    plt.show()
    plt.clf()




def collect_negative_traces(model, num_episodes, infer_method='sample'):
    reward = 0
    count = 0
    neg_traces = []
    for iter in range(0,num_episodes):
        start = random.choice(list(set(grid_locations) - set(walls)))
        start_pos = start
        # print(start_pos)
        # print("Starting point is ", start_pos)
        pick = False
        nr = False

        visit_a5 = False
        visit_c7 = False

        trace = []
        
        # print("******************", iter)
        for i in range(horizon):
            
            start_loc = pos_to_idx[start]
            grid_start = generate_starting(start)
            model.eval()
            predictions = model(grid_start.float())
            probs = Categorical(F.softmax(predictions, dim=-1))
            
            # print(F.softmax(predictions))
            if infer_method == 'sample':
                pred_action = probs.sample((1,))
            elif infer_method == 'argmax':
                pred_action = torch.argmax(F.softmax(predictions, dim=-1))[None]
            if pred_action[0] == 0:
                action_taken = 'left'
            elif pred_action[0] == 1:
                action_taken = 'right'
            elif pred_action[0] == 2:
                action_taken = 'up'
            elif pred_action[0] == 3:
                action_taken = 'down'
            else:
                action_taken = 'pick'
        
            # print(start,action_taken)
            trace.append((start, action_taken))
            if action_taken == 'left':
                if (start_loc[0]-1,start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0]-1,start_loc[1])]
                else:
                    nr = True
            elif action_taken == 'right':
                if (start_loc[0]+1,start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0]+1,start_loc[1])]
                else:
                    nr = True
            elif action_taken == 'up':
                if (start_loc[0],start_loc[1]+1) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1]+1)]
                else:
                    nr = True
            elif action_taken == 'down':
                if (start_loc[0],start_loc[1]-1) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1]-1)]
                else:
                    nr = True
            elif action_taken == 'pick':
                if (start_loc[0],start_loc[1]) in idx_to_pos.keys():
                    next_loc = idx_to_pos[(start_loc[0],start_loc[1])]
                else:
                    nr = True
            if nr or next_loc in walls:
                count += 1
                break
            
            if easy_grid:
                if next_loc == object_pos and action_taken == 'pick':
                    pick = True
                    break
            else:
                if next_loc == 'A5':
                    visit_a5 = True
                if next_loc == 'C7' and visit_a5:
                    visit_c7 = True
                    break

            start = next_loc
        
        if easy_grid:
            if pick and not nr:
                reward += 1
            else:
                neg_traces.append(trace)
        else:
            if visit_a5 and visit_c7:
                reward += 1
            else:
                neg_traces.append(trace)

    avg_reward = reward / num_episodes
    print("total average reward ", avg_reward, count)
    return neg_traces, count, avg_reward





def extract_negative_examples(neg_traces):
    faults = [expert_find_fault(trace, horizon) for trace in neg_traces]
    # print(faults)
    faults_set = set()
    for fault in faults:
        faults_set.update(fault)
    return faults_set




def train_with_negative_examples(model, faults_set):
    if len(faults_set) == 0: return
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.001)
    criterion = nn.NLLLoss()#nn.CrossEntropyLoss(reduce= False, reduction=None)

    # neg_data = [(generate_starting(state), moves[action]) for state, action in faults_set]
    neg_states = torch.stack([generate_starting(state) for state, _ in faults_set])
    neg_actions = torch.LongTensor([moves[action] for _, action in faults_set]).to(ptu.device)

    neg_dataset = GridDataset(neg_states, neg_actions)
    dataloader_neg = DataLoader(neg_dataset, batch_size=10, shuffle=False, num_workers=0)
    model.train()
    for neg_states, neg_actions in dataloader_neg:
        model.zero_grad()
        logits = -model(neg_states)

        # dot product of softmax(logits) and one_hot(action)
        # pred = F.softmax(logits, dim=1)
        # one_hot = F.one_hot(neg_actions, len(moves))
        # loss = torch.sum(pred * one_hot, dim=1).mean()

        # NLL
        pred = F.softmax(logits, dim=-1)
        loss = criterion(pred.log(), neg_actions)

        # print(pred, one_hot)
        # print(logits, neg_actions)
        # print(F.softmax(logits))
        # loss = -criterion(logits, neg_actions)
        # print(f'loss: {loss.item()}')
        
        loss.backward()
        optimizer.step()




def visualize_actions(model):
    states = np.concatenate(characters)
    actions = [torch.argmax(model(generate_starting(start))).item() for start in states]
    print(np.reshape(actions, grid_shape))




def experiment0(dir_name='exp0/'):
    global demonstrations
    num_demonstrations = [5 * i for i in range(1, 7)]
    n_iter = 10
    exp_path = data_path + dir_name

    Path(exp_path).mkdir(parents=True, exist_ok=True)
    for num in num_demonstrations[5:]:
        all_rewards = []
        for i in range(1):
            print(f'***********')
            print(f'# demo = {num}')
            print(f'***********')
            model = Network()
            model.to(ptu.device)
            rewards = []

            for itr in range(n_iter):
                # positive training
                print(f'==== Positive Iter {itr} ====')    
                demonstrations = shuffle(demonstrations)
                dataloader = get_positive_example(demonstrations[:num])
                iterations, rewards_epoch = train_with_positive_examples(model, dataloader, 100)

                rewards.append(rewards_epoch[-1])
            
            all_rewards.append(rewards)

        plot(range(n_iter), np.mean(all_rewards, axis=0), xlabel='n_iter', ylabel='reward', title=f'positive_training_n_10_b_10_demo_{num}', filename=f'{exp_path}positive_training_n_10_b_10_demo_{num}_avg')




def experiment1(dir_name='exp1/'):
    ''' 
    Experiment 1: Num pos-ex vs. num violations
    Show (averaged over 5 runs):
        1. Relation between # demo and # violations
        2. Relation between # demo and reward
        3. Relation between # demo and # unique faults

    Observations:
        1. Less trained, less reward, more violations, 

    # demo = [5, 10, 15, 20, 25, 30]
    # epoch = 50
    # sample = 5000
    '''
    global demonstrations

    all_num_violation = []
    all_rewards = []
    all_unique = []

    n_epoch = 50
    exp_path = data_path + dir_name
    Path(exp_path).mkdir(parents=True, exist_ok=True)

    for i in range(5):
        num_demonstrations = [5 * i for i in range(1, 7)]
        num_violation = []
        num_unique = []
        rewards = []
        for num in num_demonstrations:
            demonstrations = shuffle(demonstrations)
            model = Network()
            model.to(ptu.device)
            dataloader = get_positive_example(demonstrations[:num])
            iterations, rewards_epoch = train_with_positive_examples(model, dataloader, n_epoch)
            # plot(iterations, rewards_epoch)
            neg_traces, count, reward = collect_negative_traces(model, 5000, 'argmax')
            faults_set = extract_negative_examples(neg_traces)
            
            num_violation.append(count)
            rewards.append(reward)
            num_unique.append(len(faults_set))

        all_num_violation.append(num_violation)
        all_rewards.append(rewards)
        all_unique.append(num_unique)

    plot(num_demonstrations, np.mean(all_num_violation, axis=0), xlabel='# demonstrations', ylabel='# violations', title=f'demos_vs_num_violations', filename=f'{exp_path}demos_vs_num_violations')
    plot(num_demonstrations, np.mean(all_rewards, axis=0), xlabel='# demonstrations', ylabel='reward', title=f'demos_vs_reward', filename=f'{exp_path}demos_vs_reward')
    plot(num_demonstrations, np.mean(all_unique, axis=0), xlabel='# demonstrations', ylabel='# unique faults', title=f'demos_vs_num_unique_faults', filename=f'{exp_path}demos_vs_num_unique_faults')



def experiment2and3(method='argmax', dir_name='exp2/'):
    # Experiment 2 & 3: Train with different numbers of pos-ex, and retrain with neg-ex generated with argmax (or sampling) policy.
    global demonstrations
    num_demonstrations = [5 * i for i in range(1, 7)]

    n_epoch_pos = 100
    n_iter_neg = 5
    infer_method = method

    exp_path = data_path + dir_name 
    Path(exp_path).mkdir(parents=True, exist_ok=True)

    for num in num_demonstrations:
        all_num_violation = []
        all_rewards = []
        all_num_unique = []

        for i in range(5):
            print(f'***********')
            print(f'# demo = {num}')
            print(f'***********')
            demonstrations = shuffle(demonstrations)
            model = Network()
            model.to(ptu.device)
            dataloader = get_positive_example(demonstrations[:num])
            iterations, rewards_epoch = train_with_positive_examples(model, dataloader, 50)
            print('Positive training: done')
            visualize_actions(model)

            num_violation = []
            rewards = []
            num_unique = []
            
            print(f'==== Negative training start ====')
            neg_traces, count, reward = collect_negative_traces(model, 5000, infer_method)
            faults_set = extract_negative_examples(neg_traces)
            print('# unique faults = ', len(faults_set))
            num_violation.append(count)
            rewards.append(reward)
            num_unique.append(len(faults_set))

            for itr in range(n_iter_neg):
                print(f'==== Negative Iter {itr} ====')    
                train_with_negative_examples(model, faults_set)
                visualize_actions(model)

                neg_traces, count, reward = collect_negative_traces(model, 5000, infer_method)
                faults_set = extract_negative_examples(neg_traces)
                print('# unique faults = ', len(faults_set))
                num_violation.append(count)
                rewards.append(reward)
                num_unique.append(len(faults_set))


            plot(range(n_iter_neg + 1), rewards, xlabel='n_iter', ylabel='reward', title=f'{infer_method}_policy_rewards_demo_{num}', filename=f'{exp_path}{infer_method}_policy_rewards_demo_{num}')
            plot(range(n_iter_neg + 1), num_violation, xlabel='n_iter', ylabel='# negative traces', title=f'{infer_method}_policy_num_violations_demo_{num}', filename=f'{exp_path}{infer_method}_policy_num_violations_demo_{num}')
            plot(range(n_iter_neg + 1), num_unique, xlabel='n_iter', ylabel='# unique faults', title=f'{infer_method}_policy_num_unique_faults_demo_{num}', filename=f'{exp_path}{infer_method}_policy_num_unique_faults_demo_{num}')

            # neg_traces, count, reward = collect_negative_traces(model, 5000, 'sample')
            # faults_set = extract_negative_examples(neg_traces)
            # print('# unique faults SAMPLE = ', len(faults_set))
        
            all_num_violation.append(num_violation)
            all_rewards.append(rewards)
            all_num_unique.append(num_unique)

        plot(range(n_iter_neg + 1), np.mean(all_rewards, axis=0), xlabel='n_iter', ylabel='reward', title=f'{infer_method}_policy_rewards_demo_{num}', filename=f'{exp_path}{infer_method}_policy_rewards_demo_{num}_avg')
        plot(range(n_iter_neg + 1), np.mean(all_num_violation, axis=0), xlabel='n_iter', ylabel='# negative traces', title=f'{infer_method}_policy_num_violations_demo_{num}', filename=f'{exp_path}{infer_method}_policy_num_violations_demo_{num}_avg')
        plot(range(n_iter_neg + 1), np.mean(all_num_unique, axis=0), xlabel='n_iter', ylabel='# unique faults', title=f'{infer_method}_policy_num_unique_faults_demo_{num}', filename=f'{exp_path}{infer_method}_policy_num_unique_faults_demo_{num}_avg')
        




def experiment4(dir_name='exp4/'):
    # Experiment 4: Interleaving training with negative and positive examples.
    global demonstrations
    num_demonstrations = [5 * i for i in range(1, 7)]
    n_iter = 5
    infer_method = 'argmax'
    eval_method = 'argmax'

    exp_path = data_path + dir_name
    Path(exp_path).mkdir(parents=True, exist_ok=True)

    for num in num_demonstrations:
        all_rewards = []
        all_num_violation = []
        all_num_unique = []
        for i in range(5):
            print(f'***********')
            print(f'# demo = {num}')
            print(f'***********')
            model = Network()
            model.to(ptu.device)
            rewards = []
            num_violation = []
            num_unique = []

            for itr in range(n_iter):
                # positive training
                print(f'==== Positive Iter {itr*2} ====')    
                demonstrations = shuffle(demonstrations)
                dataloader = get_positive_example(demonstrations[:num])
                iterations, rewards_epoch = train_with_positive_examples(model, dataloader, 10)

                rewards.append(rewards_epoch[-1])
                print(f'itr {itr*2} (pos): {rewards_epoch[-1]}')

                
                # negative training
                neg_traces, count, reward = collect_negative_traces(model, 5000, infer_method)
                num_violation.append(count)
                faults_set = extract_negative_examples(neg_traces)
                num_unique.append(len(faults_set))
                print(f'==== Negative Iter {itr*2+1} ====')
                train_with_negative_examples(model, faults_set)
                reward = evaluate(model, 100, eval_method)
                
                rewards.append(reward)
                print(f'itr {itr*2+1} (neg): {reward}')

                neg_traces, count, reward = collect_negative_traces(model, 5000, infer_method)
                num_violation.append(count)
                faults_set = extract_negative_examples(neg_traces)
                num_unique.append(len(faults_set))
            
            all_rewards.append(rewards)
            all_num_violation.append(num_violation)
            all_num_unique.append(num_unique)

            # plot(range(2 * n_iter), rewards, xlabel='n_iter', ylabel='reward', title=f'{infer_method}_policy_rewards_demo_{num}_interleave', filename=f'{infer_method}_policy_rewards_demo_{num}_interleave')
        
        plot(range(2 * n_iter), np.mean(all_rewards, axis=0), xlabel='n_iter', ylabel='reward', title=f'interleave_{infer_method}_policy_rewards_demo_{num}', filename=f'{exp_path}interleave_{infer_method}_policy_rewards_demo_{num}_avg')
        plot(range(2 * n_iter), np.mean(all_num_violation, axis=0), xlabel='n_iter', ylabel='# violations', title=f'interleave_{infer_method}_policy_num_violation_demo_{num}', filename=f'{exp_path}interleave_{infer_method}_policy_num_violation_demo_{num}_avg')
        plot(range(2 * n_iter), np.mean(all_num_unique, axis=0), xlabel='n_iter', ylabel='# unique faults', title=f'interleave_{infer_method}_policy_num_unique_faults_demo_{num}', filename=f'{exp_path}interleave_{infer_method}_policy_num_unique_faults_demo_{num}_avg')




def combine_graphs():
    # combine two graphs
    num_demonstrations = [5 * i for i in range(1, 7)]

    pos_exp_path = data_path + 'exp0/'
    interleave_argmax_exp_path = data_path + 'exp4_argmax/'
    interleave_sample_exp_path = data_path + 'exp4_sample/'

    for num in num_demonstrations:
        pos_filename = f'positive_training_n_10_b_10_demo_{num}_avg.csv'
        interleave_argmax_filename = f'interleave_argmax_policy_rewards_demo_{num}_avg.csv'
        interleave_sample_filename = f'interleave_sample_policy_rewards_demo_{num}_avg.csv'
        df_pos = pd.read_csv(pos_exp_path + pos_filename)
        df_interleave_argmax = pd.read_csv(interleave_argmax_exp_path + interleave_argmax_filename)
        df_interleave_sample = pd.read_csv(interleave_sample_exp_path + interleave_sample_filename)

        title = f'positive_vs_interleave_demo_{num}'
        x = list(df_pos['n_iter'])
        plt.title(title)
        plt.xlabel('n_iter')
        plt.ylabel('reward')
        plt.plot(x, list(df_pos['reward']), label='positive_training')
        plt.plot(x, list(df_interleave_argmax['reward']), label='interleave_argmax')
        plt.plot(x, list(df_interleave_sample['reward']), label='interleave_sample')
        plt.legend()
        plt.savefig(data_path + title + '.png')
        plt.show()
        plt.clf()
        
experiment0()
# experiment1()
# experiment2and3('sample', 'exp3/')
# experiment4('exp4_argmax/')
# combine_graphs()
