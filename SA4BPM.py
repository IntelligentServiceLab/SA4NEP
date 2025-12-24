import os
import pickle
import random
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from utility import log_config as lg
from jinja2 import Template
import torch
from torch.utils.data import Dataset
import time
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)
seed = 42
set_seed(seed)
csv_log = 'helpdesk'
TYPE = 'all'
EPOCHS = 50
BATCH_SIZE = 16
MAX_LEN = 512
ENCODING_LENGTH = 32
LEARNING_RATE = 1e-5
HIDDEN_DIM = 384
ATTRIBUTES = ("activity", "resource", "customer", "seriousness2", "servicelevel")
DATA_ADD = "data/" + csv_log + ".csv"
def trans_to_str(val):
    if isinstance(val, float):
        new_val = str(int(val))
    elif isinstance(val, int):
        new_val = str(val)
    else:
        new_val = val
    return new_val

class CsvEventLog:
    def __init__(self, log_add, attributes=ATTRIBUTES, encoding_length=ENCODING_LENGTH):
        self.log_df = pd.read_csv(log_add)
        self.attributes = attributes
        self.encoding_length = encoding_length
        self.attributes = attributes  # list, 所选属性
        self.attributes_values = dict()
        for attribute in self.attributes:
            self.attributes_values[attribute] = list(set(self.log_df[attribute].values.tolist()))
        # prefix编码
        self.attribute_encodings = dict()
        for attribute in self.attributes:
            self.attribute_encodings[attribute] = self.random_embedding(attribute)  # 时间属性编码结果

    def random_embedding(self, attribute_name):
        """随机嵌入编码"""
        encoding_result = {}
        for val in self.attributes_values[attribute_name]:
            encoding_result[val] = np.random.rand(self.encoding_length).tolist()
        return encoding_result

    def encode_df(self, df_split, label2id_activity, time_steps, pad_value=0.0):
        df_split = df_split.sort_values('timestamp', ascending=True, kind='mergesort')
        def _left_pad_K(prefix):
            need = time_steps - len(prefix)
            if need > 0:
                pad = [
                    [[pad_value] * self.encoding_length for _ in range(len(self.attributes))]
                    for _ in range(need)
                ]
                return pad + prefix
            return prefix
        def _lookup(att, val):
            for key in (trans_to_str(val), str(val), val):
                if key in self.attribute_encodings[att]:
                    vec = self.attribute_encodings[att][key]
                    return vec if isinstance(vec, list) else np.asarray(vec, dtype=np.float32).tolist()
            return [pad_value] * self.encoding_length

        x_list, y_list = [], []
        for _, g in df_split.groupby('case', sort=False):
            g = g.sort_values('timestamp', ascending=True, kind='mergesort')
            enc, acts = [], []
            for _, row in g.iterrows():
                evt = [_lookup(att, row[att]) for att in self.attributes]
                enc.append(evt)
                acts.append(str(row['activity']))
            n = len(enc)
            if n < 2:
                continue
            #每个 case 产 n-1 个样本
            for t in range(1, n):  # 预测 acts[t]
                start = max(0, t - time_steps)
                win = enc[start:t]  # 最近 K 条，不含当前 t
                win = _left_pad_K(win)
                x_list.append(win)
                y_list.append(label2id_activity[acts[t]])
        return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)

def data_pro():
    df_train = pd.read_pickle(f'pro_data/{csv_log}_train_df.pkl')
    df_test = pd.read_pickle(f'pro_data/{csv_log}_test_df.pkl')
    # 2) 初始化属性侧，并仅用训练集语料训练
    log_attr = CsvEventLog(
        log_add=f'data/{csv_log}.csv',
        attributes=ATTRIBUTES,
        encoding_length=ENCODING_LENGTH,
    )
    # 3) 读语义侧的标签映射（activity）
    with open(f'log_history/{csv_log}/{csv_log}_label2id_{TYPE}.pkl', 'rb') as f:
        label2id_all = pickle.load(f)
    label2id_activity = label2id_all['activity']
    # 4) 按统一索引编码
    x_train, y_train = log_attr.encode_df(df_train, label2id_activity, time_steps=4)
    x_test, y_test = log_attr.encode_df(df_test, label2id_activity, time_steps=4)
    # 5) 落盘
    out = f'pro_data/{csv_log}'
    np.save(out + '_train_data_x.npy', x_train)
    np.save(out + '_train_data_y.npy', y_train)
    np.save(out + '_test_data_x.npy',  x_test)
    np.save(out + '_test_data_y.npy',  y_test)

class Log():
    def __init__(self, log, setting):
        self.__log_name = log
        self.__log = pd.read_csv('data/'+log+'.csv')
        self.__train = []
        self.__test = []
        self.__len_prefix_test = []
        self.__history_train = []
        self.__history_test = []
        self.__dict_label_train = []
        self.__dict_label_test = []
        self.__id2label = {}
        self.__label2id = {}
        self.__setting = setting
        self.__window_size = 4
        self.__max_length = 0
        self.__split_log()

    def __gen_prefix_history(self, df):
        list_seq = []
        list_len_prefix = []
        sequence = df.groupby('case', sort=False)
        event_template = Template(lg.log[self.__log_name]['event_template'])
        trace_template = Template(lg.log[self.__log_name]['trace_template'])
        dict_event_label = {v: [] for v in lg.log[self.__log_name]['event_attribute']}
        dict_len_label = {i: [] for i in range(self.__max_length)}  # 仅会用到 key=0
        for group_name, group_data in sequence:
            if len(group_data) < 2:  # 至少要有一条“下一事件”
                continue
            event_dict_hist = {}
            trace_dict_hist = {}
            ev_snips = []
            base_idx = len(list_seq)
            for index, row in group_data.iterrows():
                for v in lg.log[self.__log_name]['event_attribute']:
                    value = row[v]
                    event_dict_hist[v] = value.replace(' ', '') if isinstance(value, str) else value
                snippet = event_template.render(event_dict_hist)
                ev_snips.append(snippet)
                start = max(0, len(ev_snips) - self.__window_size)
                win_snips = ev_snips[start:]
                event_text = ' '.join(win_snips) + ' '
                for w in lg.log[self.__log_name]['trace_attribute']:
                    value = row[w]
                    trace_dict_hist[w] = value.replace(' ', '') if isinstance(value, str) else value
                trace_text = trace_template.render(trace_dict_hist)
                prefix_hist = event_text + trace_text
                list_seq.append(prefix_hist)
                eff_len = len(win_snips)
                list_len_prefix.append(eff_len)
            if len(list_seq) > base_idx:
                list_seq.pop()
                list_len_prefix.pop()
            next_act_series = group_data['activity'].shift(-1).dropna()
            dict_len_label[0].extend([self.__label2id['activity'][a] for a in next_act_series.tolist()])
            for v in lg.log[self.__log_name]['event_attribute']:
                dict_event_label[v].extend(group_data[v].shift(-1).dropna().tolist())
        return list_seq, dict_event_label, list_len_prefix, dict_len_label

    def __extract_timestamp_features(self, group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
        return group

    def __split_log(self):
        self.__log['resource'] = self.__log['resource'].astype(str)
        self.__log['resource'] = self.__log['resource'].str.replace(' ', '')
        self.__log['resource'] = self.__log['resource'].str.replace('+', '')
        self.__log['resource'] = self.__log['resource'].str.replace('-', '')
        self.__log['resource'] = self.__log['resource'].str.replace('_', '')
        self.__log.fillna('UNK', inplace=True)
        self.__log['timestamp'] = pd.to_datetime(self.__log['timestamp'])

        for c in lg.log[self.__log_name]['event_attribute']:
            if c != 'timesincecasestart':  # c!='timesincelastevent' or
                ALL_LABEL = list(self.__log[c].unique())
                self.__id2label[c] = {k: l for k, l in enumerate(ALL_LABEL)}
                self.__label2id[c] = {l: k for k, l in enumerate(ALL_LABEL)}
        cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_length = max(cont_trace)
        #按 case 生成时间相关特征
        self.__log = self.__log.groupby('case', group_keys=False).apply(self.__extract_timestamp_features)
        self.__log = self.__log.reset_index(drop=True)
        self.__log['timesincecasestart'] = (self.__log['timesincecasestart'])  # .round(3)
        self.__log['timesincecasestart'] = self.__log['timesincecasestart'].astype(int)
        #进行排序，并划分训练集和测试集
        grouped = self.__log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.7 * len(start_timestamps))]
        self.__train = self.__log[self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,
                                                                                  kind='mergesort')
        self.__test = self.__log[~self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,
                                                                                  kind='mergesort')
        os.makedirs("pro_data", exist_ok=True)
        self.__train.to_pickle(f"pro_data/{self.__log_name}_train_df.pkl")
        self.__test.to_pickle(f"pro_data/{self.__log_name}_test_df.pkl")
        self.__history_train, self.__dict_label_train, self.__len_prefix_train, dict_suffix_train = self.__gen_prefix_history(
            self.__train)
        self.__history_test, self.__dict_label_test, self.__len_prefix_test, dict_suffix_test = self.__gen_prefix_history(
            self.__test)
        for v in self.__dict_label_train:
            if v != 'timesincecasestart':
                temp_list = []
                for key in self.__dict_label_train[v]:
                    temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_train[v] = torch.tensor(temp_list, dtype=torch.long)
            else:
                self.__dict_label_train[v] = torch.tensor(self.__dict_label_train[v], dtype=torch.float32).view(-1, 1)
        for v in self.__dict_label_test:
            if v != 'timesincecasestart':
                temp_list = []
                for key in self.__dict_label_test[v]:
                    temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_test[v] = torch.tensor(temp_list, dtype=torch.long)
            else:
                self.__dict_label_test[v] = torch.tensor(self.__dict_label_test[v], dtype=torch.float32).view(-1, 1)
        os.makedirs(f'log_history/{self.__log_name}', exist_ok=True)
        self.__serialize_object(self.__history_train, 'train')
        self.__serialize_object(self.__history_test, 'test')
        self.__serialize_object(self.__dict_label_train[lg.log[self.__log_name]['target']], 'label_train')
        self.__serialize_object(self.__dict_label_test[lg.log[self.__log_name]['target']], 'label_test')

        with open('log_history/' + self.__log_name + '/' + self.__log_name + '_id2label_' + self.__setting + '.pkl',
                  'wb') as f:
            pickle.dump(self.__id2label, f)
        with open('log_history/' + self.__log_name + '/' + self.__log_name + '_label2id_' + self.__setting + '.pkl',
                  'wb') as f:
            pickle.dump(self.__label2id, f)

    def __serialize_object(self, lista, type):
        with open('log_history/' + self.__log_name + '/' + self.__log_name + '_' + type + '_' + self.__setting + '.pkl',
                  'wb') as f:
            pickle.dump(lista, f)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, attr_data=None):
        self.texts = texts
        self.labels = {}
        for v in labels:
            self.labels[v] = labels[v]
        self.tokenizer = tokenizer
        self.max_len = max_len
        if attr_data is not None:
            if not isinstance(attr_data, torch.Tensor):
                self.attr_data = torch.tensor(attr_data, dtype=torch.float)
            else:
                self.attr_data = attr_data.float()
        else:
            self.attr_data = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = {}
        for v in self.labels:
            label[v] = self.labels[v][idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }
        if self.attr_data is not None:
            item['attr_input'] = self.attr_data[idx]
        return item

class ResCell(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1):
        super(ResCell, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.stride = stride

        self.cnn1 = nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.mid_channel)
        self.cnn2 = nn.Conv2d(self.mid_channel, self.mid_channel, kernel_size=3, stride=self.stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.mid_channel)
        self.cnn3 = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.cnn4 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=self.stride)
        self.bn4 = nn.BatchNorm2d(self.out_channel)
    def forward(self, x):  # x: [32,3,73,59]
        y = F.relu(self.bn1(self.cnn1(x)))
        y = F.relu(self.bn2(self.cnn2(y)))
        y = self.bn3(self.cnn3(y))
        x = self.bn4(self.cnn4(x))
        out = F.relu(x + y)
        return out

class ResBlock(nn.Module):
    def __init__(self, att_channel, output_dim):
        super(ResBlock, self).__init__()
        self.att_channel = att_channel
        self.output_dim =  output_dim
        self.res_net1 = ResCell(self.att_channel, 64, 64, stride=2)
        self.res_net2 = ResCell(64, 128, 128, stride=2)
        self.res_net3 = ResCell(128, 256, 256, stride=2)
        self.res_net4 = ResCell(256, 512, 512, stride=2)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, self.output_dim)

    def forward(self, x):
        y1 = self.res_net1(x)  # 32,64,7,11
        y2 = self.res_net2(y1)  # 32, 128, 4, 6
        y3 = self.res_net3(y2)  # 32, 256, 2, 3
        y4 = self.res_net4(y3)  # 32, 512, 1, 2
        out = torch.squeeze(self.pooling(y4), (2, 3))  # 32, 512
        out = F.relu(self.linear(out))
        return out

def train_fn(model, train_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for X_train_batch in train_loader:
        input_ids = X_train_batch['input_ids'].to(device)
        attention_mask = X_train_batch['attention_mask'].to(device)
        attr_input = X_train_batch['attr_input'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, attr_input)
        loss = criterion['0'](output[0], X_train_batch['labels']['0'].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attr_input = batch['attr_input'].to(device)
            output = model(input_ids, attention_mask, attr_input)
            loss = criterion['0'](output[0], batch['labels']['0'].to(device))
            total += float(loss.item())
    return total / len(data_loader)

class OutputClassificationHeads(nn.Module):
    def __init__(self, gpt_model, output_sizes, attribute_num=len(ATTRIBUTES), hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gpt_model = gpt_model
        self.res_block = ResBlock(attribute_num, hidden_dim)
        self.output_layers = nn.ModuleList([
            nn.Linear(gpt_model.config.hidden_size + hidden_dim, output_sizes[i])
            for i in range(len(output_sizes))
        ])
    def forward(self, input_ids, attention_mask, attr_input):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output                      # [B, H_text]
        attr = self.res_block(attr_input.permute(0, 2, 1, 3))      # -> [B, H_attr]
        fusion_vection = torch.cat([pooled_output, attr], dim=1)   # [B, H_text+H_attr]
        out = [head(fusion_vection) for head in self.output_layers] # list of [B, C]
        return out

def _norm_case(x):
    s = str(x).strip()
    if s.lower().startswith('case '):
        s = s[5:].strip()
    try:
        return int(s)
    except:
        return s

def train_llm(model, train_data_loader, valid_data_loader, optimizer, EPOCHS, criterion, device):
    best_valid_loss = float("inf")
    early_stop_counter = 0
    patience = 5
    best_model = model
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_fn(model, train_data_loader, optimizer, device, criterion)
        valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {valid_loss:.4f}")
        if early_stop_counter >= patience:
            print("Validation loss hasn't improved for", patience, "epochs. Early stopping...")
            break
    return best_model

def train_model():
    MODEL_DIR = "models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    with open(f'log_history/{csv_log}/{csv_log}_id2label_{TYPE}.pkl', 'rb') as f:
        id2label = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_train_{TYPE}.pkl', 'rb') as f:
        train_texts = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_label_train_{TYPE}.pkl', 'rb') as f:
        y_activity_train = pickle.load(f)
    # 属性侧特征
    x_train_attr_all = np.load(f'pro_data/{csv_log}_train_data_x.npy')
    # 对齐性检查
    N = len(train_texts)
    assert x_train_attr_all.shape[0] == N, f"属性侧样本数({x_train_attr_all.shape[0]}) != 语义侧样本数({N})"
    assert len(y_activity_train) == N, f"标签长度({len(y_activity_train)}) != 样本数({N})"

    cut = int(0.7 * N)
    idx_tr = np.arange(cut)
    idx_val = np.arange(cut, N)
    train_input = [train_texts[i] for i in idx_tr]
    val_input   = [train_texts[i] for i in idx_val]
    y_all = {'0': y_activity_train.numpy() if isinstance(y_activity_train, torch.Tensor) else np.asarray(y_activity_train)}
    train_label = {k: v[idx_tr] for k, v in y_all.items()}
    val_label   = {k: v[idx_val] for k, v in y_all.items()}
    x_attr_tr  = x_train_attr_all[idx_tr]
    x_attr_val = x_train_attr_all[idx_val]
    tokenizer = AutoTokenizer.from_pretrained('utility/Bert-medium/', truncation_side='left')
    bert_backbone = AutoModel.from_pretrained('utility/Bert-medium/').to(device)
    # Dataset/Loader
    train_dataset = CustomDataset(train_input, train_label, tokenizer, MAX_LEN, attr_data=x_attr_tr)
    val_dataset   = CustomDataset(val_input,   val_label,   tokenizer, MAX_LEN, attr_data=x_attr_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    output_sizes = [len(id2label['activity'])]
    print('TRAINING START...')
    model = OutputClassificationHeads(bert_backbone, output_sizes).to(device)
    criterion = {'0': torch.nn.CrossEntropyLoss()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    startTime = time.time()
    best_model = train_llm(model, train_loader, val_loader, optimizer, EPOCHS, criterion, device)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, f"{csv_log}_{TYPE}.pth"))
    executionTime = (time.time() - startTime)
    with open(f'{csv_log}_{TYPE}.txt', 'w') as file_time:
        file_time.write(str(executionTime))
    # ======实验结果评估======
    # 1) 加载 TEST 语义文本与标签、属性侧特征
    with open(f'log_history/{csv_log}/{csv_log}_test_{TYPE}.pkl', 'rb') as f:
        test_texts = pickle.load(f)
    with open(f'log_history/{csv_log}/{csv_log}_label_test_{TYPE}.pkl', 'rb') as f:
        y_activity_test = pickle.load(f)
    x_test_attr_all = np.load(f'pro_data/{csv_log}_test_data_x.npy')
    # 2) 构建 TEST DataLoader
    y_test_map = {
        '0': y_activity_test.numpy() if isinstance(y_activity_test, torch.Tensor)
             else np.asarray(y_activity_test)
    }
    test_dataset = CustomDataset(test_texts, y_test_map, tokenizer, MAX_LEN, attr_data=x_test_attr_all)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 3)收集预测
    targets, predictions = [], []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            attr_input = batch['attr_input'].to(device)
            logits_list = best_model(input_ids, attention_mask, attr_input)   # list, 单头 -> logits_list[0]
            preds = logits_list[0].argmax(dim=1).cpu().numpy()
            labels = batch['labels']['0'].cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels)

    AVERAGING = 'weighted'  # 可选: 'macro' / 'weighted'
    def calculate_metrics(y_true, y_pred, average=AVERAGING):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'gmean': geometric_mean_score(y_true, y_pred, average=average)
        }
    metrics = calculate_metrics(targets, predictions)
    print(f"\nTEST 集综合评估指标（average={AVERAGING}）:")
    print(f"{'Accuracy':<12}: {metrics['accuracy']:.4f}")
    print(f"{'Precision':<12}: {metrics['precision']:.4f}")
    print(f"{'Recall':<12}: {metrics['recall']:.4f}")
    print(f"{'F1':<12}: {metrics['f1']:.4f}")
    print(f"{'G-mean':<12}: {metrics['gmean']:.4f}")

def main():
    Log(csv_log, TYPE)
    data_pro()
    train_model()

main()