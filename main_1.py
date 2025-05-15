# main.py  ── two-student MoE with CoT routing + weighted ensemble KL
import os, json, random, re, copy, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import datetime, timedelta
from math_utils import is_math_correct
import bitsandbytes
import torch.distributed as dist
import wandb
import pdb

import argparse, pathlib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      type=str, required=True,
                   choices=["gsm8k", "math", "arc_challenge", "anli",
                            "commonsense_qa", "date", "strategy_qa", "table_mwp"],
                   help="folder name under ./data")
    p.add_argument("--model1",    type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--model2",    type=str, default="google/gemma-7b-it")
    p.add_argument("--epochs",    type=int, default=1)
    p.add_argument("--bs",        type=int, default=4)
    p.add_argument("--max_len",   type=int, default=512)
    p.add_argument("--eval_bs",   type=int, default=8)
    p.add_argument("--eval_max_len", type=int, default=1024)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--lr_s1",     type=float, default=5e-6)
    p.add_argument("--lr_s2",     type=float, default=2e-4)
    p.add_argument("--lr_misc",   type=float, default=1e-4)
    p.add_argument("--outdir",    type=str, default="runs")
    return p.parse_args()

args = parse_args()

# build file paths from task name
root         = pathlib.Path("data") / args.task
DATA_PATH    = root / "train" / "train_cot_distill.jsonl"
DEV_PATH     = root / "train" / "test_cot_distill.jsonl"
TEST_PATH    = root / "test"  / "cot_response.jsonl"
MODEL_1      = args.model1
MODEL_2      = args.model2
NUM_EPOCHS   = args.epochs
BATCH_SIZE   = args.bs
EVAL_BATCH_SIZE = args.eval_bs
MAX_LEN      = args.max_len
EVAL_MAX_LEN = args.eval_max_len
PROJECT      = f"multi‑student‑{args.task}"
ENCODER = "FacebookAI/roberta-base"
LR_S1   = args.lr_s1          # student‑1 LoRA
LR_S2   = args.lr_s2          # student‑2 LoRA
LR_MISC = args.lr_misc        # shared modules

alpha           = 0.5     # unused (was ensemble weight in baseline)
temperature     = 1.0
kl_weight       = 0.1
USE_KL_DISTILLATION = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cuda.matmul.allow_tf32 = True      # (optional) speed

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,          # allow conditional experts
    gradient_as_bucket_view=True          # (tiny perf win)
)
pg_kwargs  = InitProcessGroupKwargs(timeout=timedelta(1800))       # long timeout for large models
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
print("local rank =", accelerator.local_process_index, "device =", accelerator.device)

automatic_wandb_config = dict(
    model_1=MODEL_1, model_2=MODEL_2, 
    encoder=ENCODER, batch_size=BATCH_SIZE,
    eval_bs=EVAL_BATCH_SIZE, num_epochs=NUM_EPOCHS,
    kl_weight=kl_weight, alpha=alpha,
    temperature=temperature, use_kl=USE_KL_DISTILLATION
)
if accelerator.is_main_process:
    wandb.init(project=PROJECT,
             name=f"{MODEL_1}_{MODEL_2}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
             config=automatic_wandb_config,
    )

# ──────────────────────── HELPERS ────────────────────────
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                data.append(json_object)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    data = data
    return data

# ──────────────────────── DATASETS ────────────────────────
class CotDataset(Dataset):
    """Return (input_ids, attn_mask, labels) where labels mask the instruction."""
    def __init__(self, data, tok, tok_1, tok_2):
        self.data, self.tok, self.tok_1, self.tok_2 = data, tok, tok_1, tok_2
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d      = self.data[idx]
        ins    = d["prompt"]
        cot    = d["messages"][1]["content"]          # teacher CoT (+ answer)
        cots = [cot, cot]


        prompt = self.tok(ins, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=512)
        L      = prompt["attention_mask"].sum()       # prompt lenth

        ids = []
        attn = []
        ids_1 = []
        attn_1 = []
        label_1 = []
        ids_2 = []
        attn_2 = []
        label_2 = []
        rand_indices = random.sample(range(len(cots)), len(cots))
        for i in range(2):
            # Randomly select a CoT from the list
            selected_cot = cots[rand_indices[i]]
            full = (self.tok(ins + selected_cot, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=512))
            full_1 = (self.tok_1(ins + selected_cot, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=MAX_LEN))
            full_2 = (self.tok_2(ins + selected_cot, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=MAX_LEN))
            
            ids.append(full["input_ids"].squeeze(0))
            attn.append(full["attention_mask"].squeeze(0))
            ids_1.append(full_1["input_ids"].squeeze(0))
            attn_1.append(full_1["attention_mask"].squeeze(0))
            ids_2.append(full_2["input_ids"].squeeze(0))
            attn_2.append(full_2["attention_mask"].squeeze(0))
            label_1.append(ids_1[i].clone())    
            label_1[i][:L]      = -100                           # ignore instruction tokens
            label_1[i][attn_1[i]==0] = -100                           # ignore padding
            label_2.append(ids_2[i].clone())
            label_2[i][:L]      = -100                           # ignore instruction tokens
            label_2[i][attn_2[i]==0] = -100                           # ignore padding

        return ids, attn, label_1, ids_1, attn_1, label_2, ids_2, attn_2

def collate(batch):
    ids, attn, lbl_1, ids_1, attn_1, lbl_2, ids_2, attn_2 = zip(*batch)
    return dict(
        input_ids     = torch.stack([torch.stack(ids) for ids in zip(*ids)]),
        attention_mask = torch.stack([torch.stack(attn) for attn in zip(*attn)]),
        input_ids_1 = torch.stack([torch.stack(ids_1) for ids_1 in zip(*ids_1)]),
        attention_mask_1 = torch.stack([torch.stack(attn_1) for attn_1 in zip(*attn_1)]),
        input_ids_2 = torch.stack([torch.stack(ids_2) for ids_2 in zip(*ids_2)]),
        attention_mask_2 = torch.stack([torch.stack(attn_2) for attn_2 in zip(*attn_2)]),
        labels_1 = torch.stack([torch.stack(lbl_1ins) for lbl_1ins in zip(*lbl_1)]),
        labels_2 = torch.stack([torch.stack(lbl_2ins) for lbl_2ins in zip(*lbl_2)]),
    )


# ──────────────────────── TOKENIZER & BASE MODEL ────────────────────────
print("Loading tokenizer & frozen backbone …")
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1)
tokenizer_1.pad_token = tokenizer_1.eos_token
tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2)
tokenizer_2.pad_token = tokenizer_2.eos_token
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_1 = AutoModelForCausalLM.from_pretrained(
    MODEL_1,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
for p in model_1.parameters(): p.requires_grad = False
model_1.eval()

model_2 = AutoModelForCausalLM.from_pretrained(
    MODEL_2,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
for p in model_2.parameters(): p.requires_grad = False
model_2.eval()

encoder_backbone = RobertaModel.from_pretrained(
    ENCODER,                      # "google-bert/bert-base-uncased"
    # quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,       # if it’s a custom repo
)

for p in encoder_backbone.parameters():
    p.requires_grad = False       # 冻结
encoder_backbone.eval().to(DEVICE)


# ──────────────────────── NEW MODULES ────────────────────────
class CotEncoder(nn.Module):
    """Frozen encoder that pools last hidden state."""
    def __init__(self, backbone, pool="mean"):
        super().__init__()
        self.backbone, self.pool = backbone, pool
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        h = self.backbone(
            input_ids     = input_ids,
            attention_mask= attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state

        if self.pool == "mean":
            num = (h * attention_mask.unsqueeze(-1)).sum(1)
            den = attention_mask.sum(1, keepdim=True)
            return (num / den).float()          # [B,d]
        return h[:,0].float()

class Router(nn.Module):
    """Two-way hard router with straight-through Gumbel-Softmax."""
    def __init__(self, d_model, tau=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 2)
        )
        self.tau = tau
    def forward(self, h):       # h:[B,d]
        return F.gumbel_softmax(self.mlp(h), tau=self.tau, hard=True)

class HiddenProjector(nn.Module):
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False, dtype=torch.float16)
    def forward(self, h):                    # h: [B,T,d]
        return self.proj(h)                  # [B,T,out_dim]

class WeightLearner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1, bias=False, dtype=torch.float16)
    def forward(self, h):       # h:[B,d]
        return torch.sigmoid(self.fc(h)).squeeze(-1)   # [B]

# ──────────────────────── LoRA STUDENTS ────────────────────────
def make_student(backbone):
    cfg   = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(backbone, cfg)
    model.print_trainable_parameters()
    return model

print("Building two LoRA students …")
student1, student2 = make_student(model_1), make_student(model_2)
# pdb.set_trace()

# ──────────────────────── INSTANTIATE NEW UTILS ────────────────────────
encoder     = CotEncoder(encoder_backbone)
router      = Router(encoder_backbone.config.hidden_size)
weight_net  = WeightLearner(student1.model.config.hidden_size)
proj1 = HiddenProjector(student1.model.config.hidden_size, student1.model.config.hidden_size).to(DEVICE)
proj2 = HiddenProjector(student2.model.config.hidden_size, student1.model.config.hidden_size).to(DEVICE)


# ──────────────────────── DATA LOADERS ────────────────────────
train_set = CotDataset(load_data(DATA_PATH), tokenizer, tokenizer_1, tokenizer_2)
# train_set = torch.utils.data.Subset(train_set, range(0, 10))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)

# ──────────────────────── OPTIMIZER ────────────────────────
opt_groups = [
    {   # LoRA adapters in student‑1
        "params": [p for p in student1.parameters() if p.requires_grad],
        "lr": LR_S1
    },
    {   # LoRA adapters in student‑2
        "params": [p for p in student2.parameters() if p.requires_grad],
        "lr": LR_S2
    },
    {   # shared modules
        "params": list(router.parameters()) + list(weight_net.parameters()),
        "lr": LR_MISC,
        "weight_decay": 0.0          # example: turn off WD just for these
    },
]
# optimizer = torch.optim.AdamW(trainables, lr=LR)
optimizer = bitsandbytes.optim.Adam8bit(opt_groups, betas=(0.9, 0.95))

# ──────────────────────── ACCELERATOR ────────────────────────
# accelerator.wait_for_everyone()

# if accelerator.is_local_main_process:
#     import pdb; pdb.set_trace()          # 只 rank‑0 停
# accelerator.wait_for_everyone() 

D = proj1.proj.out_features
(student1, student2, proj1, proj2,
router, weight_net, optimizer, train_loader) = accelerator.prepare(
    student1, student2, proj1, proj2,
    router, weight_net, optimizer, train_loader
)

encoder = encoder.to(accelerator.device)

DEVICE = accelerator.device
step=0
# ──────────────────────── TRAIN ────────────────────────
for epoch in range(1, NUM_EPOCHS + 1):
    prog = tqdm(train_loader, disable=not accelerator.is_main_process,
                desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch in prog:
        # batch 结构:
        #   input_ids         [2,B,T]  (BERT tokenizer)
        #   attention_mask    [2,B,T]
        #   input_ids_1/_2    [2,B,T]  (各自 decoder tokenizer)
        #   labels_1/_2       [2,B,T]

        # ── 先移动到 device ────────────────────────────
        step += 1
        B, T = batch["input_ids_1"].shape[1:]       # 原始 batch 尺寸
        
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # 取两条 CoT 变体 (0 / 1)
        enc_ids1, enc_mask1 = batch["input_ids"][0], batch["attention_mask"][0]
        enc_ids2, enc_mask2 = batch["input_ids"][1], batch["attention_mask"][1]

        # ── 编码 + 路由门控 ─────────────────────────────
        h1 = encoder(enc_ids1, enc_mask1)          # [B,d]
        h2 = encoder(enc_ids2, enc_mask2)

        gate1 = router(h1)                         # [B,2] one‑hot
        gate2 = router(h2)

        # 主进程广播，确保多 GPU 一致
        if dist.is_initialized():
            dist.broadcast(gate1, src=0)
            dist.broadcast(gate2, src=0)

        m1 = (gate1[:, 0].bool() | gate2[:, 0].bool())  # 选给 student1 的样本
        m2 = (gate1[:, 1].bool() | gate2[:, 1].bool())  # 选给 student2 的样本
        h1_full = torch.zeros(B, T, D, dtype=torch.float16, device=DEVICE)
        h2_full = torch.zeros_like(h1_full)

        optimizer.zero_grad()

        # ── student‑1 forward ──────────────────────────
        if m1.any():
            out1 = student1(
                input_ids     = batch["input_ids_1"][0][m1],   # 用 CoT 变体‑0
                attention_mask= batch["attention_mask_1"][0][m1],
                labels        = batch["labels_1"][0][m1],
                output_hidden_states=True
            )
            # logits1 = torch.zeros((*batch["labels_1"][0].shape,
            #                        student1.model.config.vocab_size),
            #                        dtype=out1.logits.dtype, device=DEVICE)
            # logits1[m1] = out1.logits
            hid_1  = proj1(out1.hidden_states[-1])
            h1_full[m1] = hid_1
            ce1 = out1.loss
        else:
            # logits1 = torch.zeros((*batch["labels_1"][0].shape,
            #                        student1.model.config.vocab_size),
            #                        dtype=torch.float16, device=DEVICE)
            # hid_1   = proj1(out1.hidden_states[-1])
            # h1_full[m1] = hid_1
            ce1 = torch.tensor(0., device=DEVICE)
            

        # ── student‑2 forward ──────────────────────────
        if m2.any():
            out2 = student2(
                input_ids     = batch["input_ids_2"][0][m2],
                attention_mask= batch["attention_mask_2"][0][m2],
                labels        = batch["labels_2"][0][m2],
                output_hidden_states=True
            )
            # logits2 = torch.zeros((*batch["labels_2"][0].shape,
            #                        student2.model.config.vocab_size),
            #                        dtype=out2.logits.dtype, device=DEVICE)
            # logits2[m2] = out2.logits
            hid_2   = proj2(out2.hidden_states[-1])
            h2_full[m2] = hid_2
            ce2 = out2.loss
        else:
            # logits2 = torch.zeros((*batch["labels_2"][0].shape,
            #                        student2.model.config.vocab_size),
            #                        dtype=out2.logits.dtype, device=DEVICE)
            # hid_2   = proj2(out2.hidden_states[-1])
            # h2_full[m2] = hid_2
            ce2 = torch.tensor(0., device=DEVICE)

        # ── 加权融合 + KL 蒸馏 ───────────────────────────
        tok_mask = (batch["labels_1"][0] != -100).unsqueeze(-1)
        w = weight_net((h1_full + h2_full) / 2).unsqueeze(-1)       # [B,1,1]
        logits_ens = (w.expand(h1_full.shape) * h1_full + (1 - w).expand(h2_full.shape) * h2_full).detach()
        # logits_ens = (w * h1_full[tok_mask] + (1 - w) * h2_full[tok_mask]).to(torch.bfloat16).detach()

        if USE_KL_DISTILLATION:
            # logp1 = F.log_softmax(logits1 / temperature, dim=-1)
            # logp2 = F.log_softmax(logits2 / temperature, dim=-1)
            # logpE = F.log_softmax(logits_ens / temperature, dim=-1)

            # tok_mask = (batch["labels_1"][0] != -100)
            # kl1 = F.kl_div(logp1, logpE, reduction="none",
            #                log_target=True).sum(-1)[tok_mask].mean()
            # kl2 = F.kl_div(logp2, logpE, reduction="none",
            #                log_target=True).sum(-1)[tok_mask].mean()
            if h1_full.sum() == 0 or h2_full.sum() == 0:
                kl1 = kl2 = torch.tensor(0., device=DEVICE)
            else:
                kl1 = F.mse_loss(h1_full, logits_ens)
                kl2 = F.mse_loss(h2_full, logits_ens)
            loss = ce1 + ce2 + kl_weight * (kl1 + kl2)
        else:
            kl1 = kl2 = torch.tensor(0., device=DEVICE)
            loss = ce1 + ce2

        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            wandb.log({
                "loss": loss.item(), "CE1": ce1.item(),
                "CE2": ce2.item(), "KL": (kl1 + kl2).item(),
                "epoch": epoch, "step": step
            })



        prog.set_postfix(
            loss=f"{loss.item():.4f}",
            CE1=f"{ce1.item():.3f}",
            CE2=f"{ce2.item():.3f}",
            KL=f"{(kl1+kl2).item():.3f}"
        )

# ──────────────────────── EVALUATION ────────────────────────
student1.eval(); student2.eval()
gen_student1 = accelerator.unwrap_model(student1)
gen_student2 = accelerator.unwrap_model(student2)

TEMPLATES = {
    "mistral":  "<s>[INST] Answer the following question:\n### Question:\n{question} [/INST] \n\nRead the problem carefully. Break it into small sub‑problems. Solve each sub‑problem step by step. Double-check the answers. In the final line, write only the answer inside \\boxed{{}}",
    # "mistral": "Question: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "gemma"  :  "<bos><start_of_turn>user\nAnswer the following question:\n### Question: {question}<end_of_turn>\n<start_of_turn>model\n### \n\nRead the problem carefully. Break it into small sub‑problems. Solve each sub‑problem step by step. Double-check the answers. In the final line, write only the answer inside \\boxed{{}}"  # noqa: E501
}

# --- extraction helpers --------------------------------------------------
_re_num = re.compile(
    r"""
    [-+]?                     # optional sign or dollar
    (?:                           # non‑capturing group for the integer part
        \d{1,3} (?:,\d{3})+       # 1–3 digits followed by one‑or‑more ,xxx groups
      | \d+                       # …or just digits (no commas)
    )
    (?:\.\d+)?                    # optional decimal part
    """,
    re.VERBOSE,
)
_re_choice   = re.compile(r"\(([A-F])\)")
_re_tf       = re.compile(r"\b(true|false)\b", re.I)
_re_yesno    = re.compile(r"\b(yes|no)\b", re.I)
_re_mathbox  = re.compile(r"\\boxed\{([^}]*)\}")

def parse_number(txt):
    m = re.findall(r"answer is \((\d)\)", txt)
    if m:
        return m[-1]
    else:
        m = _re_num.findall(txt)
        return m[-1].replace(",", "") if m else "N/A"

def parse_choice(txt):
    m = _re_choice.findall(txt)
    return m[-1].upper() if m else "N/A"

def parse_truefalse(txt):
    m = _re_tf.findall(txt)
    return m[-1].lower() if m else "N/A"

def parse_yesno(txt):
    m = _re_yesno.findall(txt)
    return m[-1].lower() if m else "N/A"

def parse_math_boxed(txt):
    m = _re_mathbox.findall(txt)
    return m[-1] if m else parse_number(txt)

EXTRACTORS = {
    "GSM8K"   : (parse_number, True),
    "GSM8K-Rev": (parse_number, True),
    "MATH"    : (parse_math_boxed, True),
    "TabMWP"  : (parse_math_boxed, True),
    "SQA"     : (parse_yesno, False),
    "BoolQ"   : (parse_yesno, False),
    "ANLI"    : (parse_choice, False),
    "ARC"     : (parse_choice, False),
    "CSQA"    : (parse_choice, False),
    "OBQA"    : (lambda t: parse_choice(remove_backward_answer(t)), False),
    "Date"    : (parse_choice, False),
    "ESNLI"   : (parse_choice, False),
}

# --- answer extraction helpers ------------------------------------------

def _calc_seq_logprobs(scores, seq, start_idx):
    """Sum log‑probs of generated tokens beginning at `start_idx`."""
    L, B = len(scores), seq.size(0)
    logp = torch.zeros(B, dtype=torch.float32, device=scores[0].device)
    for t in range(L):
        step_lp = scores[t].log_softmax(-1)          # (B,V)
        tok     = seq[:, start_idx + t]              # picked token ids
        logp   += step_lp.gather(1, tok.unsqueeze(1)).squeeze(1)
    return logp

class QASet(Dataset):
    """(question, gold_answer) records loaded from `path`."""
    def __init__(self, path): self.data = load_data(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return d["question"], d["answer"]

def evaluate_dual_students(
    path: str,
    task_name: str = "GSM8K",
    model_tag: str = "mistral",
    batch_size: int = 8,
    max_new: int = MAX_LEN,
):
    extract, is_math   = EXTRACTORS[task_name]
    template           = TEMPLATES[model_tag]
    loader             = DataLoader(QASet(path), batch_size=batch_size, shuffle=False)

    s1_ok = s2_ok = ensemble_ok = total = 0
    for qs, gold in tqdm(loader, desc="Eval‑dual", dynamic_ncols=True):
        # build identical prompts for both students
        prompts = [template.format(question=q) for q in qs]

        # encode separately for each tokenizer
        enc1 = tokenizer_1(prompts, return_tensors="pt", padding=True).to(DEVICE)
        enc2 = tokenizer_2(prompts, return_tensors="pt", padding=True).to(DEVICE)
        p_len1 = enc1.attention_mask.sum(-1)   # prompt lengths (B,)
        p_len2 = enc2.attention_mask.sum(-1)

        # generate with scores
        out1 = gen_student1.generate(**enc1, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer_1.pad_token_id,
                                     return_dict_in_generate=True, output_scores=True)
        out2 = gen_student2.generate(**enc2, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer_2.pad_token_id,
                                     return_dict_in_generate=True, output_scores=True)

        # decode + extract answers
        preds1 = [extract(o) for o in tokenizer_1.batch_decode(out1.sequences,
                                                              skip_special_tokens=True)]
        preds2 = [extract(o) for o in tokenizer_2.batch_decode(out2.sequences,
                                                              skip_special_tokens=True)]

        # sequence‑level log‑probabilities
        lp1 = _calc_seq_logprobs(out1.scores, out1.sequences, int(p_len1.min())).cpu()
        lp2 = _calc_seq_logprobs(out2.scores, out2.sequences, int(p_len2.min())).cpu()

        for p1, p2, l1, l2, g in zip(preds1, preds2, lp1, lp2, gold):
            best = p1 if l1 >= l2 else p2

            # per‑student tallies
            if is_math:
                s1_ok += is_math_correct(p1, g)
                s2_ok += is_math_correct(p2, g)
                ensemble_ok += is_math_correct(best, g)
            else:
                s1_ok += int(p1 == g)
                s2_ok += int(p2 == g)
                ensemble_ok += int(best == g)
            total += 1

        # if accelerator.is_main_process:
        wandb.log({
            "eval/acc_overall": ensemble_ok / total,
            "eval/acc_s1":      s1_ok / total,
            "eval/acc_s2":      s2_ok / total,
        })
        print(f"\nOverall acc: {ensemble_ok/total:.4f}")
        print(f"Student‑1 acc: {s1_ok/total:.4f}")
        print(f"Student‑2 acc: {s2_ok/total:.4f}")

    # if accelerator.is_main_process:
    print(f"\nOverall  acc: {ensemble_ok/total:.4f}")
    print(f"Student‑1 acc: {s1_ok/total:.4f}")
    print(f"Student‑2 acc: {s2_ok/total:.4f}")

    return ensemble_ok / total

# ─── run it ───────────────────────────────────────────────────
if accelerator.is_main_process:
    evaluate_dual_students(TEST_PATH, task_name="GSM8K", model_tag="mistral")