def LSTM_AI_train():
    """
    策略：
    1. 文本预编码后 → 一个巨 tensor，一次 .to(device)，永不回 CPU
    2. 训练时 CPU 只做 torch.randint + GPU index_select（极轻量）
    3. 不用 Dataset / DataLoader —— 消除 Python 迭代器开销
    4. 大 batch + 大 hidden_size 填满 Tensor Core
    5. 4 层 LSTM + 残差 + LayerNorm + GeLU + Dropout
    6. cuDNN benchmark / TF32 / AMP / fused AdamW 全开
    7. 余弦退火 + 梯度裁剪 + 最佳模型保存
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import os, time

    # ======================== 0. 全局 GPU 优化开关 ========================
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        DEVICE = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        print(f'GPU: {gpu_name} ({gpu_mem} GB VRAM)  |  '
              f'CUDNN benchmark={torch.backends.cudnn.benchmark}')
    else:
        DEVICE = torch.device('cpu')

    # ======================== 1. 读取 & 编码到 GPU ========================
    with open('龙族Ⅰ-Ⅳ.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    c2i['<|e|>'] = 0
    i2c = {v: k for k, v in c2i.items()}
    vocab_size = len(c2i)
    print(f'文本长度: {len(text):,}  词汇表: {vocab_size}')

    # ★ 关键优化：全量编码 → GPU，永不下车
    t0 = time.perf_counter()
    encoded = torch.tensor([c2i[c] for c in text], dtype=torch.long, device=DEVICE)
    print(f'编码 + 上传 GPU: {time.perf_counter() - t0:.2f}s  '
          f'({encoded.element_size() * encoded.numel() / 1024**2:.1f} MB)')

    # ======================== 2. 超参 ========================
    SEQ_LEN = 256
    BATCH_SIZE = 256                # 大 batch 喂满 GPU
    EMB_SIZE = 384
    HIDDEN_SIZE = 768               # 大 hidden 占满 Tensor Core
    NUM_LAYERS = 4
    DROPOUT = 0.15
    LEARNING_RATE = 3e-3
    EPOCHS = 20
    EVAL_ITERS = 50
    GRAD_CLIP = 1.0

    # ======================== 3. 模型 ========================
    class DeepLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, EMB_SIZE)
            self.emb_drop = nn.Dropout(0.1)
            self.proj_in = nn.Linear(EMB_SIZE, HIDDEN_SIZE)

            # 4 层 LSTM
            self.lstm = nn.LSTM(
                input_size=HIDDEN_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT if NUM_LAYERS > 1 else 0,
                batch_first=True,
            )

            self.ln = nn.LayerNorm(HIDDEN_SIZE)
            self.drop = nn.Dropout(DROPOUT)

            # 2 层 MLP lm_head（比单 Linear 更强）
            self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 2)
            self.fc2 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
            self.lm_head = nn.Linear(HIDDEN_SIZE, vocab_size)
            self.act = nn.GELU()

            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        def forward(self, x):
            # x: (B, T) 已在 GPU
            emb = self.emb_drop(self.emb(x))               # (B, T, EMB)
            h = self.proj_in(emb)                          # (B, T, H)

            lstm_out, _ = self.lstm(h)                     # (B, T, H)
            h = self.drop(self.ln(lstm_out + h))            # 残差 + LN + Drop

            # 2 层 MLP + 残差 lm_head
            skip = h
            h = self.act(self.fc1(h))
            h = self.drop(h)
            h = self.act(self.fc2(h))
            h = self.drop(h)
            h = h + skip                                   # (B, T, H)

            return self.lm_head(h)                         # (B, T, V)

    model = DeepLSTM().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total_params/1e6:.1f}M  '
          f'模型 ~{total_params*4/1024**2:.0f} MB (fp32)')

    # ======================== 4. GPU 端 batch 构造 ========================
    def get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        """
        从 GPU 常驻 encoded 中直接 index_select 组装 batch。
        CPU 只生成随机起始索引（微秒级），其余全在 GPU 内部完成。
        """
        n = len(encoded)
        start_idx = torch.randint(0, n - seq_len - 1, (batch_size,), device=DEVICE)
        idx = (start_idx.unsqueeze(1)
               + torch.arange(seq_len, device=DEVICE).unsqueeze(0))
        x = encoded[idx]                              # (B, T) — 纯 GPU 索引
        y = encoded[idx + 1]                          # (B, T) — 右移一位
        return x, y

    # ======================== 5. 损失估算 ========================
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1))
            losses.append(loss.item())
        model.train()
        return torch.tensor(losses).mean().item()

    print(f'初始 loss: {estimate_loss():.4f}')

    # ======================== 6. 训练 ========================
    scaler = torch.amp.GradScaler('cuda')

    # fused AdamW: 比普通 AdamW 更快更省显存
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=0.1, betas=(0.9, 0.95),
        fused=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01)

    total_tokens = len(encoded) - SEQ_LEN - 1
    steps_per_epoch = total_tokens // (BATCH_SIZE * SEQ_LEN)
    print(f'Steps/epoch ≈ {steps_per_epoch:,}  ({total_tokens:,} tokens)')

    best_loss = float('inf')
    best_path = 'best_model_lstm.pth'
    t_all = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t_ep = time.perf_counter()

        for step in range(steps_per_epoch):
            x, y = get_batch()                           # GPU 直接取

            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()

        avg_train = epoch_loss / steps_per_epoch
        test_loss = estimate_loss()
        lr_now = scheduler.get_last_lr()[0]

        print(f'[epoch {epoch+1:>3}/{EPOCHS}]  '
              f'train={avg_train:.4f}  test={test_loss:.4f}  '
              f'lr={lr_now:.6f}  '
              f'{time.perf_counter()-t_ep:.1f}s'
              f'{"最佳!" if test_loss < best_loss else ""}')

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'model_type': 'lstm',
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_loss': best_loss,
                'vocab_size': vocab_size,
                'c2i': c2i, 'i2c': i2c,
                'config': {
                    'SEQ_LEN': SEQ_LEN,
                    'EMB_SIZE': EMB_SIZE,
                    'HIDDEN_SIZE': HIDDEN_SIZE,
                    'NUM_LAYERS': NUM_LAYERS,
                    'DROPOUT': DROPOUT,
                },
            }, best_path)

        scheduler.step()

    print(f'\n总耗时: {time.perf_counter()-t_all:.0f}s  '
          f'最佳 test_loss: {best_loss:.4f}')

    # ======================== 7. 推理 ========================
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])

    @torch.no_grad()
    def generate(start_str=' ', max_new=800,
                 temperature=0.85, top_k=80, top_p=0.92):
        encoded_start = [c2i.get(c, 0) for c in start_str]
        pad_len = SEQ_LEN - len(encoded_start)
        pad_id = c2i.get(' ', 0)
        ctx = torch.tensor(
            [pad_id] * pad_len + encoded_start, device=DEVICE).unsqueeze(0)

        model.eval()
        out = encoded_start.copy()
        for _ in range(max_new):
            logits = model(ctx)[:, -1, :] / temperature

            # top-k
            if top_k > 0:
                v, __ = torch.topk(logits, min(top_k, vocab_size))
                logits[logits < v[:, -1:]] = float('-inf')
            # top-p
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                c = torch.cumsum(F.softmax(s, -1), -1)
                mask = c > top_p
                mask[:, 1:] = mask[:, :-1].clone(); mask[:, 0] = False
                logits[mask.scatter(1, si, mask)] = float('-inf')

            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            ctx = torch.cat((ctx[:, 1:], nxt), dim=-1)
            tok = nxt.item()
            out.append(tok)
            if tok == 0:  # <|e|>
                break

        return ''.join(i2c[t] for t in out)

    print('\n' + '═' * 60)
    print('生成结果:')
    print('═' * 60)
    print(generate('夜色'))

def GPT_train_AI():
    """
    GPT 风格 Decoder-only Transformer，针对《龙族》风格小说生成优化。

    为什么 Transformer 比 LSTM/RNN 更适合这个任务：
    ┌──────────────┬─────────────────────┬──────────────────────┐
    │              │  LSTM/RNN           │  Transformer (GPT)   │
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ 长程依赖      │ 梯度消失，~50步遗忘  │ 自注意力：全局上下文    │
    │ 训练并行度    │ 串行（慢）           │ 并行（快）            │
    │ 小说连贯性    │ 差                   │ 好（事实上所有 LLM     │
    │              │                     │  都用这个架构)        │
    │ 实现复杂度    │ 简单                 │ 中等                 │
    └──────────────┴─────────────────────┴──────────────────────┘

    设计要点：
    1. Decoder-only (GPT-2 风格)：因果注意力，自回归生成
    2. Pre-LayerNorm：训练更稳定
    3. RoPE 旋转位置编码：比可学习位置编码外推性更好
    4. 自动启用 Flash Attention（RTX 5060 Blackwell 架构原生支持）
    5. 权重共享 (weight tying)：embedding 和 lm_head 共享权重，省参数
    6. 数据全量常驻 GPU，与 LSTM_GPU_MAX 相同策略
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import os, time, math

    # ======================== 0. GPU 优化开关 ========================
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        DEVICE = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        print(f'GPU: {gpu_name} ({gpu_mem} GB VRAM)')
    else:
        DEVICE = torch.device('cpu')

    # ======================== 1. 读取 & 编码到 GPU ========================
    with open('龙族Ⅰ-Ⅳ.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    c2i['<|e|>'] = 0
    i2c = {v: k for k, v in c2i.items()}
    vocab_size = len(c2i)
    print(f'文本: {len(text):,} 字符  词汇表: {vocab_size}')

    t0 = time.perf_counter()
    encoded = torch.tensor([c2i[c] for c in text], dtype=torch.long, device=DEVICE)
    print(f'编码→GPU: {time.perf_counter()-t0:.2f}s  '
          f'({encoded.element_size()*encoded.numel()/1024**2:.1f} MB)')

    # ======================== 2. 超参 ========================
    SEQ_LEN = 256
    BATCH_SIZE = 128                    # Transformer 比 LSTM 更吃显存
    N_EMBD = 576                        # 嵌入维度（能被 head 数整除）
    N_HEAD = 9                          # 注意力头数 (576/9=64 per head)
    N_LAYER = 10                        # Transformer 层数
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4                # Transformer 对 lr 敏感，用小一些
    EPOCHS = 15
    EVAL_ITERS = 50
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 500                  # lr warmup

    # ======================== 3. RoPE 旋转位置编码 ========================
    def precompute_rope_freqs(dim, seq_len, theta=10000.0):
        """预计算 RoPE 频率表"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)                    # (seq_len, dim/2)
        return torch.cat((freqs, freqs), dim=-1)          # (seq_len, dim)  — cos, sin 共用

    def apply_rope(x, rope_freqs):
        """
        对输入应用 RoPE。x: (B, n_head, T, head_dim)
        rope_freqs: (T, head_dim)
        """
        T = x.shape[2]
        cos = torch.cos(rope_freqs[:T]).unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = torch.sin(rope_freqs[:T]).unsqueeze(0).unsqueeze(0)

        d2 = x.shape[-1] // 2
        x_rot = torch.stack((-x[..., d2:], x[..., :d2]), dim=-1)
        x_rot = x_rot.reshape_as(x)

        return x * cos + x_rot * sin

    # 预计算到 GPU
    rope_freqs = precompute_rope_freqs(N_EMBD // N_HEAD, SEQ_LEN * 2).to(DEVICE)

    # ======================== 4. Transformer 模型 ========================
    class CausalSelfAttention(nn.Module):
        """多头因果自注意力 + RoPE"""
        def __init__(self):
            super().__init__()
            self.n_head = N_HEAD
            self.head_dim = N_EMBD // N_HEAD
            assert self.head_dim * N_HEAD == N_EMBD, "N_EMBD 必须能被 N_HEAD 整除"

            # QKV 合并为一个矩阵乘法（更高效）
            self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
            self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
            self.attn_drop = nn.Dropout(DROPOUT)
            self.proj_drop = nn.Dropout(DROPOUT)

        def forward(self, x):
            B, T, C = x.shape

            # QKV 投影 + 拆分为多头
            qkv = self.qkv(x)                                  # (B, T, 3*C)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # RoPE（仅在 Q、K 上）
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

            # Flash Attention（PyTorch 2.0+ 自动选择最优实现）
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=DROPOUT if self.training else 0.0,
                is_causal=True,
            )                                                  # (B, nh, T, hd)

            y = y.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)
            y = self.proj_drop(self.proj(y))
            return y

    class MLP(nn.Module):
        """前馈网络：GELU + 4x 扩展"""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
            self.fc2 = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)
            self.drop = nn.Dropout(DROPOUT)

        def forward(self, x):
            return self.drop(self.fc2(F.gelu(self.fc1(x))))

    class TransformerBlock(nn.Module):
        """Pre-LN Transformer 块"""
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(N_EMBD)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(N_EMBD)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.ln1(x))   # Pre-LN: 先归一化再注意力
            x = x + self.mlp(self.ln2(x))    # Pre-LN: 先归一化再 FFN
            return x

    class GPT(nn.Module):
        """GPT 风格 Decoder-only Transformer"""
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
            self.drop = nn.Dropout(DROPOUT)
            self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYER)])
            self.ln_f = nn.LayerNorm(N_EMBD)     # 最终 LayerNorm

            # lm_head 不单独分配权重：与 tok_emb 共享（weight tying）
            self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

            # 权重初始化
            self.apply(self._init_weights)
            # 特殊处理残差投影，防止深层梯度爆炸
            for pn, p in self.named_parameters():
                if pn.endswith('proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        def forward(self, x):
            # x: (B, T)
            x = self.drop(self.tok_emb(x))          # (B, T, C)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return self.lm_head(x)                   # (B, T, V)

            # 注意：如果启用 weight tying，lm_head.weight 就是 tok_emb.weight

    model = GPT().to(DEVICE)

    # ====== Weight Tying: embedding 和 lm_head 共享权重 ======
    model.lm_head.weight = model.tok_emb.weight       # 节省 vocab_size * N_EMBD 参数

    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total_params/1e6:.1f}M  '
          f'模型 ~{total_params*4/1024**2:.0f} MB (fp32)')
    print(f'层数: {N_LAYER}  注意力头: {N_HEAD}  嵌入维度: {N_EMBD}  '
          f'序列长度: {SEQ_LEN}')

    # ======================== 5. GPU 端 batch 构造 ========================
    def get_batch(batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        n = len(encoded)
        start_idx = torch.randint(0, n - seq_len - 1, (batch_size,), device=DEVICE)
        idx = (start_idx.unsqueeze(1)
               + torch.arange(seq_len, device=DEVICE).unsqueeze(0))
        x = encoded[idx]
        y = encoded[idx + 1]
        return x, y

    # ======================== 6. 损失估算 ========================
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1))
            losses.append(loss.item())
        model.train()
        return torch.tensor(losses).mean().item()

    print(f'初始 loss: {estimate_loss():.4f}  (理论最小值 ≈ -ln(1/{vocab_size}) = '
          f'{-math.log(1/vocab_size):.2f})')

    # ======================== 7. 训练 ========================
    scaler = torch.amp.GradScaler('cuda')

    # AdamW + 小 weight_decay（Transformer 不需要大 weight_decay）
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=0.01, betas=(0.9, 0.95),
        fused=True,
    )

    total_tokens = len(encoded) - SEQ_LEN - 1
    steps_per_epoch = total_tokens // (BATCH_SIZE * SEQ_LEN)
    total_steps = steps_per_epoch * EPOCHS
    print(f'Steps/epoch ≈ {steps_per_epoch:,}  总 steps ≈ {total_steps:,}')

    # 余弦退火 + 线性 warmup
    def get_lr(step):
        """线性 warmup → 余弦衰减"""
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return LEARNING_RATE * 0.01 + (LEARNING_RATE - LEARNING_RATE * 0.01) * \
               0.5 * (1.0 + math.cos(math.pi * progress))

    best_loss = float('inf')
    best_path = 'best_model_gpt.pth'
    t_all = time.perf_counter()
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t_ep = time.perf_counter()

        for step in range(steps_per_epoch):
            # 手动调整 lr（warmup + cosine）
            lr_now = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

            x, y = get_batch()

            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

        avg_train = epoch_loss / steps_per_epoch
        test_loss = estimate_loss()

        print(f'[epoch {epoch+1:>3}/{EPOCHS}]  '
              f'train={avg_train:.4f}  test={test_loss:.4f}  '
              f'lr={lr_now:.6f}  '
              f'{time.perf_counter()-t_ep:.1f}s'
              f'{"  ★ 最佳!" if test_loss < best_loss else ""}')

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'model_type': 'gpt',
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_loss': best_loss, 'vocab_size': vocab_size,
                'c2i': c2i, 'i2c': i2c,
                'config': {
                    'SEQ_LEN': SEQ_LEN, 'N_EMBD': N_EMBD,
                    'N_HEAD': N_HEAD, 'N_LAYER': N_LAYER,
                    'DROPOUT': DROPOUT,
                },
            }, best_path)

    print(f'\n总耗时: {time.perf_counter()-t_all:.0f}s  '
          f'最佳 test_loss: {best_loss:.4f}')

    # ======================== 8. 推理 ========================
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])

    @torch.no_grad()
    def generate(start_str='夜色', max_new=1000,
                 temperature=0.85, top_k=80, top_p=0.92):
        encoded_start = [c2i.get(c, 0) for c in start_str]
        pad_len = SEQ_LEN - len(encoded_start)
        pad_id = c2i.get(' ', 0)
        ctx = torch.tensor(
            [pad_id] * pad_len + encoded_start, device=DEVICE).unsqueeze(0)

        model.eval()
        out = encoded_start.copy()
        for _ in range(max_new):
            # 截断过长上下文
            if ctx.shape[1] > SEQ_LEN:
                ctx = ctx[:, -SEQ_LEN:]

            logits = model(ctx)[:, -1, :] / temperature

            if top_k > 0:
                v, __ = torch.topk(logits, min(top_k, vocab_size))
                logits[logits < v[:, -1:]] = float('-inf')
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                c = torch.cumsum(F.softmax(s, -1), -1)
                mask = c > top_p
                mask[:, 1:] = mask[:, :-1].clone(); mask[:, 0] = False
                logits[mask.scatter(1, si, mask)] = float('-inf')

            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            ctx = torch.cat((ctx, nxt), dim=-1)
            tok = nxt.item()
            out.append(tok)
            if tok == 0:
                break

        return ''.join(i2c[t] for t in out)

    print('\n' + '═' * 60)
    print('生成结果:')
    print('═' * 60)
    print(generate('夜色'))

def test__():
    """通用测试函数：自动识别 LSTM/GPT 模型并生成文本。
    配置区只需修改 MODEL_PATH 和生成参数即可。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import numpy as np

    # ======================== 配置 ========================
    MODEL_PATH = 'best_model_gpt.pth'   # 模型路径
    START_TEXT = '龙族'                   # 起始文本
    MAX_NEW_TOKENS = 2000                # 最大生成长度
    TEMPERATURE = 0.85                   # 温度
    TOP_K = 80                           # top-k
    TOP_P = 0.92                         # top-p
    SEED = 33550336                       # 随机种子（None=不固定）
    # ======================================================

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {DEVICE}')

    # ── 种子 ──
    def _seed_to_int(s):
        if isinstance(s, int): return s
        h = 0
        for ch in s: h = (h * 31 + ord(ch)) & 0xFFFF_FFFF
        return h if h < 2**31 else h - 2**32

    if SEED is not None:
        sv = _seed_to_int(SEED)
        random.seed(sv); np.random.seed(sv); torch.manual_seed(sv)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(sv); torch.cuda.manual_seed_all(sv)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'随机种子: "{SEED}" → {sv}')
    else:
        print('随机种子: 未设置')

    # ── 加载 checkpoint ──
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model_type = ckpt.get('model_type', 'unknown')
    vocab_size = ckpt['vocab_size']
    c2i = ckpt['c2i']
    i2c = ckpt['i2c']
    cfg = ckpt.get('config', {})
    print(f'模型类型: {model_type}  '
          f'词汇表: {vocab_size}  '
          f'test_loss: {ckpt.get("test_loss", "?"):.4f}')

    # ── 根据 model_type 重建模型 ──
    if model_type == 'lstm':
        seq_len = cfg.get('SEQ_LEN', 256)
        emb_size = cfg.get('EMB_SIZE', 384)
        hidden_size = cfg.get('HIDDEN_SIZE', 768)
        num_layers = cfg.get('NUM_LAYERS', 4)
        dropout = cfg.get('DROPOUT', 0.15)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_size)
                self.emb_drop = nn.Dropout(0.1)
                self.proj_in = nn.Linear(emb_size, hidden_size)
                self.lstm = nn.LSTM(
                    input_size=hidden_size, hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True)
                self.ln = nn.LayerNorm(hidden_size)
                self.drop = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
                self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                self.act = nn.GELU()

            def forward(self, x):
                emb = self.emb_drop(self.emb(x))
                h = self.proj_in(emb)
                lstm_out, _ = self.lstm(h)
                h = self.drop(self.ln(lstm_out + h))
                skip = h
                h = self.act(self.fc1(h)); h = self.drop(h)
                h = self.act(self.fc2(h)); h = self.drop(h)
                return self.lm_head(h + skip)

    elif model_type == 'gpt':
        seq_len = cfg.get('SEQ_LEN', 256)
        n_embd = cfg.get('N_EMBD', 576)
        n_head = cfg.get('N_HEAD', 9)
        n_layer = cfg.get('N_LAYER', 10)
        dropout = cfg.get('DROPOUT', 0.1)

        head_dim = n_embd // n_head

        def precompute_rope_freqs(dim, sl, theta=10000.0):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(sl, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            return torch.cat((freqs, freqs), dim=-1)

        def apply_rope(x, rope_f):
            T = x.shape[2]
            cos = torch.cos(rope_f[:T]).unsqueeze(0).unsqueeze(0)
            sin = torch.sin(rope_f[:T]).unsqueeze(0).unsqueeze(0)
            d2 = x.shape[-1] // 2
            x_rot = torch.stack((-x[..., d2:], x[..., :d2]), dim=-1).reshape_as(x)
            return x * cos + x_rot * sin

        rope_freqs = precompute_rope_freqs(head_dim, seq_len * 2).to(DEVICE)

        class CausalSelfAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.nh = n_head; self.hd = head_dim
                self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
                self.proj = nn.Linear(n_embd, n_embd, bias=False)
                self.attn_drop = nn.Dropout(dropout)
                self.proj_drop = nn.Dropout(dropout)

            def forward(self, x):
                B, T, C = x.shape
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
                k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
                v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
                q = apply_rope(q, rope_freqs)
                k = apply_rope(k, rope_freqs)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None,
                    dropout_p=dropout if self.training else 0.0, is_causal=True)
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.proj_drop(self.proj(y))

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
                self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
                self.drop = nn.Dropout(dropout)
            def forward(self, x):
                return self.drop(self.fc2(F.gelu(self.fc1(x))))

        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(n_embd)
                self.attn = CausalSelfAttention()
                self.ln2 = nn.LayerNorm(n_embd)
                self.mlp = MLP()
            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_emb = nn.Embedding(vocab_size, n_embd)
                self.drop = nn.Dropout(dropout)
                self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(n_embd)
                self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
                # weight tying
                self.lm_head.weight = self.tok_emb.weight

            def forward(self, x):
                x = self.drop(self.tok_emb(x))
                for block in self.blocks:
                    x = block(x)
                return self.lm_head(self.ln_f(x))

    else:
        raise ValueError(f'未知模型类型: {model_type}，请在 checkpoint 的 model_type 字段指定 "lstm" 或 "gpt"')

    model = Model().to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total/1e6:.1f}M  模型已加载\n')

    # ── 生成 ──
    @torch.no_grad()
    def generate(start_str, max_new=800, temperature=0.85, top_k=80, top_p=0.92):
        encoded_start = [c2i.get(c, 0) for c in start_str]
        pad_len = seq_len - len(encoded_start)
        if pad_len < 0:
            # 起始文本过长则截断
            encoded_start = encoded_start[-seq_len:]
            pad_len = 0
        pad_id = c2i.get(' ', 0)
        ctx = torch.tensor(
            [pad_id] * pad_len + encoded_start, device=DEVICE).unsqueeze(0)

        out = encoded_start.copy()
        for _ in range(max_new):
            if ctx.shape[1] > seq_len:
                ctx = ctx[:, -seq_len:]
            logits = model(ctx)[:, -1, :] / temperature

            if top_k > 0:
                v, __ = torch.topk(logits, min(top_k, vocab_size))
                logits[logits < v[:, -1:]] = float('-inf')
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                c = torch.cumsum(F.softmax(s, -1), -1)
                mask = c > top_p
                mask[:, 1:] = mask[:, :-1].clone(); mask[:, 0] = False
                logits[mask.scatter(1, si, mask)] = float('-inf')

            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            ctx = torch.cat((ctx, nxt), dim=-1)
            tok = nxt.item()
            out.append(tok)
            if tok == 0:
                break

        return ''.join(i2c[t] for t in out)

    print('═' * 60)
    result = generate(START_TEXT, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P)
    print(result)
    print('═' * 60)
    print(f'共 {len(result)} 字符')

test__()