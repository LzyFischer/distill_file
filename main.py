# main.py  ── two-student MoE with CoT routing + weighted ensemble KL
import os, json, random, re, copy, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import datetime, timedelta
import torch.distributed as dist


import pdb

# ──────────────────────── CONFIG ────────────────────────
DATA_PATH       = "data/gsm8k/train/train_cot_distill.jsonl"
DEV_PATH        = "data/gsm8k/train/test_cot_distill.jsonl"
TEST_PATH       = "data/gsm8k/test/cot_response.jsonl"
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.1"
BATCH_SIZE      = 4
NUM_EPOCHS      = 3
MAX_LEN = 1024 
LR              = 1e-4

alpha           = 0.5     # unused (was ensemble weight in baseline)
temperature     = 1.0
kl_weight       = 0.5
USE_KL_DISTILLATION = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cuda.matmul.allow_tf32 = True      # (optional) speed

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,          # allow conditional experts
    gradient_as_bucket_view=True          # (tiny perf win)
)
pg_kwargs  = InitProcessGroupKwargs(timeout=timedelta(1800))       # long timeout for large models
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
print("local rank =", accelerator.local_process_index, "device =", accelerator.device)

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
    data = data[:100]
    return data

# ──────────────────────── DATASETS ────────────────────────
class ReasoningDataset(Dataset):
    """
    Returns: instruction, two sampled CoTs, label
    The two CoTs are later routed to two students.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['prompt']
        cot = item['messages'][1]['content']
        cot1, cot2 = [cot, cot]
        return instruction, cot1, cot2, None

class EvalDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return d["question"], d["answer"]

# ──────────────────────── TOKENIZER & BASE MODEL ────────────────────────
print("Loading tokenizer & frozen backbone …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    # device_map="auto",
    trust_remote_code=True,
)
for p in base_model.parameters(): p.requires_grad = False
base_model.eval()

# ──────────────────────── COLLATE FN ────────────────────────
def collate_fn(batch):
    ins, cot1, cot2, _ = zip(*batch)                 # we don't need labels here
    out = {k: [] for k in [
        "enc_ids1","enc_mask1","enc_ids2","enc_mask2",
        "gen_ids","gen_mask","labels1","labels2"
    ]}
                    # keep every sequence the same length

    for instr, cot_a, cot_b in zip(ins, cot1, cot2):
        # ─ full sequences: instruction + CoT ─
        seq1 = tokenizer(instr + cot_a,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_LEN)
        seq2 = tokenizer(instr + cot_b,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_LEN)

        # ─ prompt only (needed for generation + to locate the cut-off L) ─
        prompt = tokenizer(instr,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_LEN)

        L = prompt["attention_mask"].sum().item()    # true prompt length (no padding)

        # ─ encoder inputs ─
        enc_ids1, enc_mask1 = seq1["input_ids"], seq1["attention_mask"]
        enc_ids2, enc_mask2 = seq2["input_ids"], seq2["attention_mask"]

        # ─ labels: clone input_ids and mask prompt + padding ─
        labels1 = enc_ids1.clone()
        labels2 = enc_ids2.clone()
        pad_id = tokenizer.pad_token_id
        labels1[:, :L] = -100
        labels2[:, :L] = -100
        labels1.masked_fill_(labels1 == pad_id, -100)
        labels2.masked_fill_(labels2 == pad_id, -100)

        # ─ collect tensors ─
        for k, v in zip(
            ["enc_ids1", "enc_mask1", "enc_ids2", "enc_mask2",
            "gen_ids",  "gen_mask",  "labels1",  "labels2"],
            [enc_ids1,   enc_mask1,   enc_ids2,   enc_mask2,
            prompt["input_ids"], prompt["attention_mask"],
            labels1, labels2]):
            out[k].append(v)

    # ─ stack lists into batch tensors ─
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    return out

# ──────────────────────── NEW MODULES ────────────────────────
class CotEncoder(nn.Module):
    """Frozen encoder that pools last hidden state."""
    def __init__(self, backbone, pool="mean"):
        super().__init__()
        self.backbone, self.pool = backbone, pool
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        h = self.backbone.model(
            input_ids     = input_ids,
            attention_mask= attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states[-1]                     # [B,T,d]
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

class WeightLearner(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
    def forward(self, h):       # h:[B,d]
        return torch.sigmoid(self.fc(h)).squeeze(-1)   # [B]

# ──────────────────────── LoRA STUDENTS ────────────────────────
def make_student(backbone):
    model = copy.deepcopy(backbone)
    cfg   = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

print("Building two LoRA students …")
student1, student2 = make_student(base_model), make_student(base_model)
# pdb.set_trace()

# ──────────────────────── INSTANTIATE NEW UTILS ────────────────────────
encoder     = CotEncoder(base_model)
router      = Router(base_model.config.hidden_size)
weight_net  = WeightLearner(base_model.config.hidden_size)


# ──────────────────────── DATA LOADERS ────────────────────────
train_set = ReasoningDataset(load_data(DATA_PATH), tokenizer)
# train_set = torch.utils.data.Subset(train_set, range(0, 10))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)

# ──────────────────────── OPTIMIZER ────────────────────────
trainables = (list(student1.parameters()) +
              list(student2.parameters()) +
              list(router.parameters())   +
              list(weight_net.parameters()))
optimizer = torch.optim.AdamW(trainables, lr=LR)

# ──────────────────────── ACCELERATOR ────────────────────────
(student1, student2, 
router, weight_net, optimizer, train_loader) = accelerator.prepare(
    student1, student2,
    router, weight_net, optimizer, train_loader
)

encoder = encoder.to(accelerator.device)

DEVICE = accelerator.device
# ──────────────────────── TRAIN ────────────────────────
for epoch in range(1, NUM_EPOCHS+1):
    prog = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for batch in prog:
        # ─ move to device ─
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        gen_ids, labels1, labels2 = batch["gen_ids"], batch["labels1"], batch["labels2"]

        # ─ encode both CoTs ─
        h1 = encoder(batch["enc_ids1"], batch["enc_mask1"])
        h2 = encoder(batch["enc_ids2"], batch["enc_mask2"])

        # ─ routing decisions ─
        gate1 = router(h1)          # [B,2] one-hot
        gate2 = router(h2)
        
        if dist.is_initialized():
            gate1 = gate1.clone()
            gate2 = gate2.clone()
            dist.broadcast(gate1, src=0)
            dist.broadcast(gate2, src=0)

        m1 = (gate1[:, 0].bool() | gate2[:, 0].bool())
        m2 = (gate1[:, 1].bool() | gate2[:, 1].bool())

        optimizer.zero_grad()

        VOCAB_SIZE = base_model.config.vocab_size
        # ─ student-1 forward ─
        if m1.any():
            pdb.set_trace()
            out1 = student1(input_ids=gen_ids[m1], labels=labels1[m1])
            logits1 = torch.zeros((*labels1.shape, VOCAB_SIZE),
                                  dtype=out1.logits.dtype, device=DEVICE)
            logits1[m1] = out1.logits
            ce1 = out1.loss
        else:
            logits1 = torch.zeros((*labels1.shape, VOCAB_SIZE),
                                  dtype=torch.float16, device=DEVICE)
            ce1 = torch.tensor(0., device=DEVICE)

        # ─ student-2 forward ─
        if m2.any():
            out2 = student2(input_ids=gen_ids[m2], labels=labels2[m2])
            logits2 = torch.zeros_like(logits1)
            logits2[m2] = out2.logits
            ce2 = out2.loss
        else:
            logits2 = torch.zeros_like(logits1)
            ce2 = torch.tensor(0., device=DEVICE)

        # ─ learn sample weight w ∈ (0,1) ─
        w = weight_net((h1 + h2) / 2).view(-1,1,1)     # [B,1,1]
        logits_ens = w * logits1 + (1 - w) * logits2
        logits_ens = logits_ens.to(torch.bfloat16).detach()

        # ─ KL divergence against ensemble ─
        if USE_KL_DISTILLATION:
            logp1 = F.log_softmax(logits1 / temperature, dim=-1)
            logp2 = F.log_softmax(logits2 / temperature, dim=-1)
            logpE = F.log_softmax(logits_ens.detach() / temperature, dim=-1)

            tok_mask = (labels1 != -100)
            kl1 = F.kl_div(logp1, logpE, reduction="none",
                           log_target=True).sum(-1)[tok_mask].mean()
            kl2 = F.kl_div(logp2, logpE, reduction="none",
                           log_target=True).sum(-1)[tok_mask].mean()
            loss = ce1 + ce2 + kl_weight * (kl1 + kl2)
        else:
            kl1 = kl2 = torch.tensor(0., device=DEVICE)
            loss = ce1 + ce2
        accelerator.backward(loss)
        optimizer.step()

        prog.set_postfix({
            "loss": f"{loss.item():.4f}",
            "CE1":  f"{ce1.item():.3f}",
            "CE2":  f"{ce2.item():.3f}",
            "KL":   f"{(kl1+kl2).item():.3f}"
        })

# ──────────────────────── SIMPLE EVAL (router → one student) ────────────────────────
student1.eval(), student2.eval()
gen_student1 = accelerator.unwrap_model(student1)
gen_student2 = accelerator.unwrap_model(student2)

def extract_number(text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else None

@torch.no_grad()
def evaluate(dataset_path):
    data = load_data(dataset_path)
    loader = DataLoader(EvalDataset(data), batch_size=1)
    correct_1 = correct_2 = correct = total = 0
    for ins, label in tqdm(loader, desc=f"Eval {os.path.basename(dataset_path)}"):
        prompt = ins[0]
        e = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        h = encoder(e["input_ids"], e["attention_mask"])
        gate = router(h)[0]              # [2]
        if gate[0] == 1:
            outs = gen_student1.generate(**e, max_length=256)
        else:
            outs = gen_student2.generate(**e, max_length=256)

        outs_1 = gen_student1.generate(**e, max_length=256)
        outs_2 = gen_student2.generate(**e, max_length=256)
        pred_ans_1 = extract_number(tokenizer.decode(outs_1[0], skip_special_tokens=True))
        pred_ans_2 = extract_number(tokenizer.decode(outs_2[0], skip_special_tokens=True))
        pred_ans = extract_number(tokenizer.decode(outs[0], skip_special_tokens=True))

        gold_ans = extract_number(label[0])
        correct += (pred_ans == gold_ans)
        correct_1 += (pred_ans_1 == gold_ans)
        correct_2 += (pred_ans_2 == gold_ans)
        total   += 1
        if total % 10 == 0:
            print(f"Accuracy: {correct / total:.4f}")
            print(f"Accuracy 1: {correct_1 / total:.4f}")
            print(f"Accuracy 2: {correct_2 / total:.4f}")
    acc = correct / total
    print(f"Accuracy: {acc:.4f}")

if accelerator.is_main_process:
    evaluate(TEST_PATH)