import os
import random
import time
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm



# Set the root directory containing all language subfolders
DATA_ROOT = "/multilingual_transliteration/aksharantar_sampled"
SAVE_PATH = "/multilingual_transliteration/seq2seq_attn_multilingual_best.pt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EMB_DIM = 256
HID_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_BIDIR = True
DROPOUT = 0.2
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
CLIP = 1.0
BEAM_SIZE = 5
TEACHER_FORCING_START = 0.9
TEACHER_FORCING_END = 0.4
USE_LANG_TOKEN = True


try:
    from google.colab import drive
    drive.mount('/content/drive')
    print(f"Google Drive mounted. Checking path: {DATA_ROOT}")

    # Check if the path actually exists after mounting
    if not os.path.isdir(DATA_ROOT):
        print(f"WARNING: The directory {DATA_ROOT} does not exist.")
        print("Please check your DATA_ROOT path. Common issues:")
        print("1. A typo in the path (it's case-sensitive).")
        print("2. The folder 'aksharantar_sampled' is not in your 'MyDrive' folder.")
    else:
        print("Data directory found. Proceeding with loading.")

except ImportError:
    print("Not in Google Colab, assuming local file system. Skipping drive mount.")
    if not os.path.isdir(DATA_ROOT):
         print(f"WARNING: The local directory {DATA_ROOT} does not exist.")


# ---------- Load & merge all language files ----------

def load_and_preprocess_csv(filepath, lang_code):
    """
    Loads a single CSV file (with NO header) and applies preprocessing.
    Assumes:
    - Column 0: input_text (independent variable)
    - Column 1: target_text (dependent variable)
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found, skipping: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, header=None)
    except pd.errors.EmptyDataError:
        print(f"Warning: Skipping {filepath}, file is empty.")
        return None
    except Exception as e:
        print(f"Warning: Failed to read {filepath}. Error: {e}")
        return None

    if 0 not in df.columns or 1 not in df.columns:
         print(f"Warning: Skipping {filepath}, expected at least 2 columns, but found {len(df.columns)}.")
         return None

    df = df.rename(columns={0: 'input_text', 1: 'target_text'})

    if 'input_text' not in df.columns or 'target_text' not in df.columns:
        print(f"Warning: Skipping {filepath}, failed to rename columns.")
        return None

    df = df[['input_text','target_text']].astype(str).copy()
    df['language'] = lang_code

    if USE_LANG_TOKEN:
        lang_token = f"<{lang_code}>"
        df['input_text'] = lang_token + " " + df['input_text'].astype(str)

    return df

# Find all language subdirectories in the DATA_ROOT
if not os.path.isdir(DATA_ROOT):
    raise FileNotFoundError(f"Data root directory not found. Please mount Google Drive and check your DATA_ROOT path: {DATA_ROOT}")

lang_codes = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
# lang_codes=lang_codes[:2]
print(f"Found {len(lang_codes)} language folders: {lang_codes}")

train_dfs, valid_dfs, test_dfs = [], [], []

# Loop through each language folder and load its train, valid, and test files
for lang in tqdm(lang_codes, desc="Loading all language data"):
    train_file = os.path.join(DATA_ROOT, lang, f"{lang}_train.csv")
    valid_file = os.path.join(DATA_ROOT, lang, f"{lang}_valid.csv")
    test_file  = os.path.join(DATA_ROOT, lang, f"{lang}_test.csv")

    # Load and append training data
    df_train_lang = load_and_preprocess_csv(train_file, lang)
    if df_train_lang is not None:
        train_dfs.append(df_train_lang)

    # Load and append validation data
    df_valid_lang = load_and_preprocess_csv(valid_file, lang)
    if df_valid_lang is not None:
        valid_dfs.append(df_valid_lang)

    # Load and append test data
    df_test_lang = load_and_preprocess_csv(test_file, lang)
    if df_test_lang is not None:
        test_dfs.append(df_test_lang)

# Concatenate all dataframes from all languages
if not train_dfs or not valid_dfs or not test_dfs:
    raise ValueError("No data was loaded. Check your DATA_ROOT path and file structure. Did you mount Google Drive?")

train_df = pd.concat(train_dfs, ignore_index=True)
valid_df = pd.concat(valid_dfs, ignore_index=True)
test_df  = pd.concat(test_dfs, ignore_index=True)

print("Successfully loaded and merged all language files.")

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Train size:", len(train_df), "Valid size:", len(valid_df), "Test size:", len(test_df))


def build_vocab(series):
    chars = sorted(set("".join(series.astype(str))))
    # Add special tokens (in this order) so indices are predictable
    itos = ['<pad>','<sos>','<eos>','<unk>'] + chars
    stoi = {c:i for i,c in enumerate(itos)}
    return stoi, itos

# Build vocab from training set
src_stoi, src_itos = build_vocab(train_df['input_text'])
tgt_stoi, tgt_itos = build_vocab(train_df['target_text'])

SRC_PAD = src_stoi['<pad>']; TGT_PAD = tgt_stoi['<pad>']
TGT_SOS = tgt_stoi['<sos>']; TGT_EOS = tgt_stoi['<eos>']

print("Vocab sizes (src/tgt):", len(src_itos), len(tgt_itos))
print(f"Sample src token '<asm>': {src_stoi.get('<asm>')}")
print(f"Sample src token 'a': {src_stoi.get('a')}")
print(f"Sample tgt token 'a': {tgt_stoi.get('a')}")


# ---------- Dataset & Collate ----------
class TransliterationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.src_list = df['input_text'].astype(str).tolist()
        self.tgt_list = df['target_text'].astype(str).tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    def __len__(self): return len(self.src_list)
    def encode_src(self, s):
        return torch.tensor([self.src_vocab.get(ch, self.src_vocab['<unk>']) for ch in s], dtype=torch.long)
    def encode_tgt(self, t):
        return torch.tensor([self.tgt_vocab.get(ch, self.tgt_vocab['<unk>']) for ch in t], dtype=torch.long)
    def __getitem__(self, idx):
        return self.encode_src(self.src_list[idx]), self.encode_tgt(self.tgt_list[idx])

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=SRC_PAD)
    tgts_with_tokens = [torch.cat([torch.tensor([TGT_SOS]), t, torch.tensor([TGT_EOS])]) for t in tgts]
    tgt_pad = pad_sequence(tgts_with_tokens, batch_first=True, padding_value=TGT_PAD)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgts_with_tokens], dtype=torch.long)
    return src_pad, src_lens, tgt_pad, tgt_lens

train_loader = DataLoader(TransliterationDataset(train_df, src_stoi, tgt_stoi), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(TransliterationDataset(valid_df, src_stoi, tgt_stoi), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TransliterationDataset(test_df, src_stoi, tgt_stoi), batch_size=512, shuffle=False, collate_fn=collate_fn)

# ---------- Model ----------
def make_rnn(cell, emb_dim, hid_dim, layers, bidir=False, dropout=0.0):
    if cell == "lstm":
        return nn.LSTM(emb_dim, hid_dim, num_layers=layers, batch_first=True, bidirectional=bidir, dropout=dropout if layers>1 else 0)
    elif cell == "gru":
        return nn.GRU(emb_dim, hid_dim, num_layers=layers, batch_first=True, bidirectional=bidir, dropout=dropout if layers>1 else 0)
    else:
        return nn.RNN(emb_dim, hid_dim, num_layers=layers, batch_first=True, nonlinearity='tanh', dropout=dropout if layers>1 else 0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, bidir=True, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=SRC_PAD)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidir = bidir
        self.rnn = make_rnn("lstm", emb_dim, hid_dim, n_layers, bidir, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, (h_n, c_n) = self.rnn(emb)
        return outputs, (h_n, c_n)

class LuongAttention(nn.Module):
    def __init__(self, hid_dim, bidirectional_encoder=True):
        super().__init__()
        enc_mult = 2 if bidirectional_encoder else 1
        self.Wa = nn.Linear(hid_dim, hid_dim * enc_mult, bias=False)
    def forward(self, dec_hidden, enc_outputs, mask):
        proj = self.Wa(dec_hidden)
        proj = proj.unsqueeze(1)
        scores = torch.bmm(proj, enc_outputs.transpose(1,2)).squeeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, enc_bidirectional=True, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=TGT_PAD)
        enc_mult = 2 if enc_bidirectional else 1
        self.rnn = nn.LSTM(emb_dim + hid_dim * enc_mult, hid_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.attn = LuongAttention(hid_dim, bidirectional_encoder=enc_bidirectional)
        self.fc_out = nn.Linear(hid_dim + hid_dim * enc_mult + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
    def forward(self, input_tok, last_hidden, last_cell, enc_outputs, mask):
        emb = self.dropout(self.embedding(input_tok))     
        dec_h_top = last_hidden[-1]                         
        context, attn = self.attn(dec_h_top, enc_outputs, mask)  
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)  
        output, (h_n, c_n) = self.rnn(rnn_input, (last_hidden, last_cell))
        output = output.squeeze(1)                       
        fc_in = torch.cat([output, context, emb.squeeze(1)], dim=1)  
        prediction = self.fc_out(fc_in)                   
        return prediction, h_n, c_n, attn

class Bridge(nn.Module):
    def __init__(self, enc_layers, dec_layers, enc_hid, dec_hid, bidirectional=True):
        super().__init__()
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.bid = bidirectional
        enc_mult = 2 if bidirectional else 1
        self.hid_map = nn.Linear(enc_hid * enc_mult, dec_hid)
        self.cell_map = nn.Linear(enc_hid * enc_mult, dec_hid)
    def forward(self, h_enc, c_enc):
        enc_layers = self.enc_layers
        enc_mult = 2 if self.bid else 1
        B = h_enc.size(1)
        h_enc = h_enc.view(enc_layers, enc_mult, B, -1)
        c_enc = c_enc.view(enc_layers, enc_mult, B, -1)
        h_cat = h_enc.view(enc_layers, B, -1)
        c_cat = c_enc.view(enc_layers, B, -1)
        h_list = []
        c_list = []
        for i in range(self.dec_layers):
            idx = min(i, enc_layers-1)
            h_proj = torch.tanh(self.hid_map(h_cat[idx]))
            c_proj = torch.tanh(self.cell_map(c_cat[idx]))
            h_list.append(h_proj.unsqueeze(0))
            c_list.append(c_proj.unsqueeze(0))
        h_dec = torch.cat(h_list, dim=0)
        c_dec = torch.cat(c_list, dim=0)
        return h_dec, c_dec

# Instantiate models
enc = Encoder(len(src_itos), EMB_DIM, HID_DIM, ENC_LAYERS, bidir=ENC_BIDIR, dropout=DROPOUT).to(DEVICE)
dec = DecoderWithAttention(len(tgt_itos), EMB_DIM, HID_DIM, DEC_LAYERS, enc_bidirectional=ENC_BIDIR, dropout=DROPOUT).to(DEVICE)
bridge = Bridge(ENC_LAYERS, DEC_LAYERS, HID_DIM, HID_DIM, bidirectional=ENC_BIDIR).to(DEVICE)

def count_params(*mods):
    return sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)
print("Trainable params:", count_params(enc, dec, bridge))

# ---------- Training & Eval (unchanged logic) ----------
def make_mask(src_pad):
    return (src_pad != SRC_PAD).to(DEVICE)

def train_epoch(model_components, loader, optimizer, epoch):
    enc, dec, bridge = model_components
    enc.train(); dec.train(); bridge.train()
    total_loss = 0
    n_batches = 0
    tf_ratio = TEACHER_FORCING_START + (TEACHER_FORCING_END - TEACHER_FORCING_START) * (epoch / max(1, NUM_EPOCHS-1))
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD)
    for src, src_lens, tgt, tgt_lens in tqdm(loader, desc=f"Train E{epoch}"):
        src = src.to(DEVICE); tgt = tgt.to(DEVICE)
        mask = make_mask(src)
        optimizer.zero_grad()
        enc_outputs, (h_n, c_n) = enc(src)
        h_dec, c_dec = bridge(h_n, c_n)
        max_tgt_len = tgt.size(1)
        input_tok = tgt[:,0].unsqueeze(1)
        losses = []
        for t in range(1, max_tgt_len):
            pred, h_dec, c_dec, attn = dec(input_tok, h_dec, c_dec, enc_outputs, mask)
            target = tgt[:,t]
            losses.append(criterion(pred, target))
            teacher_force = random.random() < tf_ratio
            top1 = pred.argmax(1).unsqueeze(1)
            input_tok = tgt[:,t].unsqueeze(1) if teacher_force else top1
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()) + list(bridge.parameters()), CLIP)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)

def evaluate_components(model_components, loader):
    enc, dec, bridge = model_components
    enc.eval(); dec.eval(); bridge.eval()
    total_loss = 0.0; n_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD)
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in tqdm(loader, desc="Valid"):
            src = src.to(DEVICE); tgt = tgt.to(DEVICE)
            mask = make_mask(src)
            enc_outputs, (h_n, c_n) = enc(src)
            h_dec, c_dec = bridge(h_n, c_n)
            input_tok = tgt[:,0].unsqueeze(1)
            losses = []
            for t in range(1, tgt.size(1)):
                pred, h_dec, c_dec, attn = dec(input_tok, h_dec, c_dec, enc_outputs, mask)
                target = tgt[:,t]
                losses.append(criterion(pred, target))
                input_tok = pred.argmax(1).unsqueeze(1)
            if losses:
                loss = torch.stack(losses).mean()
                total_loss += loss.item(); n_batches += 1
    return total_loss / max(1, n_batches)

# Beam search decode (char-level)
def beam_search_decode(enc, dec, bridge, src_sentence, beam_size=BEAM_SIZE, max_len=80):
    enc.eval(); dec.eval(); bridge.eval()
    with torch.no_grad():
        src_idx = torch.tensor([[src_stoi.get(c, src_stoi['<unk>']) for c in src_sentence]]).to(DEVICE)
        enc_outputs, (h_n, c_n) = enc(src_idx)
        mask = (src_idx != SRC_PAD).to(DEVICE)
        h_dec, c_dec = bridge(h_n, c_n)
        init = ([TGT_SOS], 0.0, h_dec, c_dec)
        beam = [init]
        completed = []
        for _ in range(max_len):
            new_beam = []
            for tokens, logp, h_cur, c_cur in beam:
                if tokens[-1] == TGT_EOS:
                    completed.append((tokens, logp))
                    continue
                input_tok = torch.tensor([[tokens[-1]]], device=DEVICE)
                pred, h_new, c_new, attn = dec(input_tok, h_cur, c_cur, enc_outputs, mask)
                log_probs = torch.log_softmax(pred, dim=1).squeeze(0)
                topk_logp, topk_idx = torch.topk(log_probs, beam_size)
                for l, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    new_tokens = tokens + [idx]
                    new_logp = logp + l
                    new_beam.append((new_tokens, new_logp, h_new, c_new))
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            if not beam:
                break
        candidates = completed + beam
        if not candidates:
             return "" # Handle empty candidates
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        best_tokens = candidates[0][0]
        toks = []
        for t in best_tokens:
            if t == TGT_SOS: continue
            if t == TGT_EOS: break
            toks.append(tgt_itos[t])
        return "".join(toks)

# ---------- Run training ----------
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(bridge.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

best_val_loss = float('inf')
patience = 4
no_improve = 0

print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    start = time.time()
    train_loss = train_epoch((enc, dec, bridge), train_loader, optimizer, epoch)
    val_loss = evaluate_components((enc, dec, bridge), valid_loader)
    scheduler.step(val_loss)
    elapsed = time.time() - start
    print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, time {elapsed:.1f}s")
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        no_improve = 0
        torch.save({
            'enc': enc.state_dict(),
            'dec': dec.state_dict(),
            'bridge': bridge.state_dict(),
            'src_itos': src_itos, 'src_stoi': src_stoi,
            'tgt_itos': tgt_itos, 'tgt_stoi': tgt_stoi
        }, SAVE_PATH)
        print(f"Saved best checkpoint to {SAVE_PATH}")
    else:
        no_improve += 1
        print(f"No improvement {no_improve}/{patience}")
        if no_improve >= patience:
            print("Early stopping.")
            break
print("--- Training Finished ---")

# ---------- Inference & Evaluation ----------
print(f"\n--- Loading best model for evaluation ---")
if not os.path.exists(SAVE_PATH):
    print("Error: Saved model not found. Skipping evaluation.")
    # Set examples to reflect codes, but exit gracefully
    examples = ["<tam> namaskaram", "<hin> pani", "<ben> prem"]
    print("\nExamples (ensure language codes match folder names):")
    for w in examples:
        print(w, "->", "(Model not trained or loaded)")
else:
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    enc.load_state_dict(ckpt['enc'])
    dec.load_state_dict(ckpt['dec'])
    bridge.load_state_dict(ckpt['bridge'])
    # Ensure vocabs are loaded from checkpoint for consistency
    src_stoi = ckpt['src_stoi']
    src_itos = ckpt['src_itos']
    tgt_stoi = ckpt['tgt_stoi']
    tgt_itos = ckpt['tgt_itos']

    enc.to(DEVICE); dec.to(DEVICE); bridge.to(DEVICE)
    print("Model loaded successfully.")

    # Evaluate on test using beam search
    def levenshtein(a, b):
        n, m = len(a), len(b)
        if n == 0: return m
        if m == 0: return n
        dp = list(range(m+1))
        for i in range(1, n+1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m+1):
                cur = dp[j]
                if a[i-1] == b[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = cur
        return dp[m]

    total_chars = 0
    correct_chars = 0
    total_edits = 0
    total_ref_chars = 0

    # Store some examples
    results_df = []

    for i in tqdm(range(len(test_df)), desc="Test eval"):
        src = test_df.iloc[i]['input_text']
        ref = test_df.iloc[i]['target_text']
        pred = beam_search_decode(enc, dec, bridge, src, beam_size=BEAM_SIZE, max_len=80)

        if i < 10: # Store first 10 examples
             results_df.append({'Source': src, 'Reference': ref, 'Prediction': pred})

        L = min(len(pred), len(ref))
        correct_chars += sum(pred[j]==ref[j] for j in range(L))
        total_chars += L
        total_edits += levenshtein(pred, ref)
        total_ref_chars += len(ref)

    # Calculate Character Accuracy (based on min length)
    char_acc = correct_chars / total_chars if total_chars>0 else 0.0

    # Calculate Character Error Rate (CER)
    cer = total_edits / total_ref_chars if total_ref_chars>0 else None

    # Calculate Word Accuracy (Exact Match)
    exact_matches = 0
    for i in tqdm(range(len(test_df)), desc="Word Accuracy"):
        src = test_df.iloc[i]['input_text']
        ref = test_df.iloc[i]['target_text']
        pred = beam_search_decode(enc, dec, bridge, src, beam_size=BEAM_SIZE, max_len=80)
        if pred == ref:
            exact_matches += 1

    word_acc = exact_matches / len(test_df) if len(test_df) > 0 else 0.0

    print(f"\n--- Test Set Metrics ---")
    print(f"Word Accuracy (Exact Match): {word_acc:.4f} ({exact_matches}/{len(test_df)})")
    print(f"Character Error Rate (CER):  {cer:.4f}")
    print(f"Character Accuracy (approx): {char_acc:.4f}")

    print("\n--- Test Examples (from test set) ---")
    print(pd.DataFrame(results_df).to_string())

    # MODIFIED: Quick examples (use lang codes from folders, e.g., 'tam', 'hin', 'ben')
    examples = ["<tam> namaskaram", "<hin> pani", "<ben> prem", "<asm> val"]
    print("\n--- Custom Examples (beam) ---")
    for w in examples:
        if not all(c in src_stoi for c in w):
            print(f"{w} -> (Skipped: contains chars not in training vocab)")
            continue
        print(w, "->", beam_search_decode(enc, dec, bridge, w, beam_size=BEAM_SIZE))