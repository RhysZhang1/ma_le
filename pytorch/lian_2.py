def LSTM_1():
    # 长短期记忆网络
    # 隐藏状态--分-->细胞状态，隐藏状态
    # 细胞状体：长期记忆，隐藏状态：短期记忆
    # 门控组件：进行信息的筛选：遗忘门，输入门，输出门
    # 备选细胞状态：更新细胞状态

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import os

    os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_root = 'C:\\Users\\rhys1\\Desktop\\zane_project\\python\\pytorch\\hf_cache'

    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = cache_root + r'\datasets'  # 处理好的 Arrow 数据集
    os.environ["HF_MODULES_CACHE"] = cache_root + r'\modules'  # 加载脚本(.py)
    os.environ["HF_HUB_CACHE"] = cache_root + r'\hub'  # 原始下载文件

    from datasets import load_dataset

    raw_datasets = load_dataset("code_search_net", "python")
    datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])

    print(datasets[8]['whole_func_string'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    learning_rate = 0.001
    eval_iters = 10
    batch_size = 1000
    sequence_len = 64


    class CharTokenizer:  # 分词器
        def __init__(self, data, end_ind=0):
            # data: list[str]
            chars = sorted(list(set(''.join(data))))
            self.char2ind = {s: i + 1 for i, s in enumerate(chars)}  # 加2是预留的特殊字符
            # 正向
            self.char2ind['<|e|>'] = end_ind  # 表示字符串结尾
            self.ind2char = {v: k for k, v in self.char2ind.items()}  # 反向
            self.end_ind = end_ind

        def encode(self, x):
            # x:list[str]
            return [self.char2ind[i] for i in x]

        def decode(self, x):
            # x:int or list[x]
            if isinstance(x, int):
                return self.ind2char[x]
            return [self.ind2char[i] for i in x]


    tokenizer = CharTokenizer(datasets['whole_func_string'])

    test_str = 'def f(x):'
    print(tokenizer.encode(test_str))
    print(''.join(tokenizer.decode(range(len(tokenizer.char2ind)))))


    def process(data, tokenizer, sequence_len=sequence_len):
        text = data['whole_func_string']
        # text:list[str]
        inputs, labels = [], []
        for t in text:
            enc = tokenizer.encode(t)
            enc += [tokenizer.end_ind]
            # enc = [tokenizer.begin_ind] * (sequence_len - 1) + enc + [tokenizer.end_ind]
            for i in range(len(enc) - sequence_len):  # 没有办法处理太短的文本
                inputs.append(enc[i:i + sequence_len])
                labels.append(enc[i + 1:i + 1 + sequence_len])
        return {'inputs': inputs, 'labels': labels}


    tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
    f = lambda x: process(x, tokenizer)
    tokenized = tokenized.map(f, batched=True, remove_columns=datasets.column_names)
    tokenized.set_format(type='torch', device=device)

    train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)


    @torch.no_grad()
    def generate(model, context, tokenizer, max_new_token=300):
        # context:(1,T)
        out = context.tolist()[0]
        model.eval()
        for i in range(max_new_token):
            # 可以考虑截断背景，使得文本生成更加贴近训练
            # logits=model(context[:,-sequence_len:])
            logits = model(context)  # (1,T,98)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (1,98)
            # 随机生成文本
            ix = torch.multinomial(probs, num_samples=1)  # (1,1)
            # 更新背景
            context = torch.concat((context, ix), dim=-1)
            out.append(ix.item())
            if out[-1] == tokenizer.end_ind:
                break
        model.train()
        return out


    def estimate_loss(model):
        re={}
        model.eval()
        re['train']=_loss(model,train_loader)
        re['test']=_loss(model,test_loader)
        model.train()
        return re

    @torch.no_grad()
    def _loss(model,data_loader):
        loss=[]
        data_iter=iter(data_loader)
        for k in range(eval_iters):
            data=next(data_iter,None)
            if data is None:
                data_iter=iter(data_loader)
                data=next(data_iter,None)
            inputs,labels=data['inputs'],data['labels']     #(B,T)
            logits=model(inputs)                            #(B,T,vs)
            loss.append(F.cross_entropy(logits.transpose(-2,-1),labels).item())
        return torch.tensor(loss).mean().item()

    def train_model(model,optimizer,epochs=10):
        lossi=[]
        for epoch in range(epochs):
            for i,data in enumerate(train_loader,0):
                inputs,labels=data['inputs'],data['labels']   #(B,T)
                optimizer.zero_grad()
                logits=model(inputs)                      #(B,T,vs)
                loss=F.cross_entropy(logits.transpose(-2,-1),labels)
                lossi.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#梯度裁剪
                optimizer.step()
            stats=estimate_loss(model)
            train_loss=f'train loss {stats["train"]:.4f}'
            test_loss=f'test loss {stats["test"]:.4f}'
            print(f'epoch{epoch:>2} : {train_loss} , {test_loss}')
        return lossi

    class LSTMCell(nn.Module):
        def __init__(self, input_size, hidden_size):#假设输出C和隐藏H的大小一样
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            combine_size=hidden_size+input_size
            self.forget_gate=nn.Linear(combine_size,hidden_size) #遗忘门
            self.in_gate=nn.Linear(combine_size,hidden_size)  #输入门
            self.out_gate=nn.Linear(combine_size,hidden_size) #输出门
            self.new_cell_gate=nn.Linear(combine_size,hidden_size)  #更新细胞状态

        def forward(self,input,state=None):
            #input:(B,I)
            #state:((B,H),(B,H))
            B=input.shape[0]
            if state is None:   #初始化
                state=self.init_state(B,input.device)
            hs,cs=state
            combine=torch.concat((input,hs),dim=-1)  #张量拼接   (B,I+H)
            #细胞状态更新
            ingate=F.sigmoid(self.in_gate(combine))
            forgetgate=F.sigmoid(self.forget_gate(combine))
            ncs=F.tanh(self.new_cell_gate(combine))
            cs=(cs*forgetgate)+(ingate*ncs)
            #隐藏状态更新
            outgate=F.sigmoid(self.out_gate(combine))
            hs=F.tanh(cs)*outgate
            return hs,cs

        def init_state(self,B,device):
            hs=torch.zeros((B,self.hidden_size),device=device)
            cs=torch.zeros((B,self.hidden_size),device=device)
            return hs,cs

    l_cell=LSTMCell(3,4)
    x=torch.randn(5,3)
    a,b=l_cell(x)
    print(a.shape,b.shape)

    class LSTM(nn.Module):
        def __init__(self,input_size,hidden_size):
            super().__init__()
            self.cell=LSTMCell(input_size,hidden_size=hidden_size)

        def forward(self,input,state=None):
            #input:(B,T,C)
            #state:((B,H),(B,H))
            #out:(B,T,H)
            B,T,C=inputs.shape
            re=[]
            for i in range(T):
                state=self.cell(input[:,i,:],state)
                re.append(state[0])
            return torch.stack(re,dim=1)               #(B,T,H)


LSTM_1()