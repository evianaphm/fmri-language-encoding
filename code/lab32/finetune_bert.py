# code/lab32/finetune_bert.py
#
# Full fine-tuning of BERT for fMRI encoding (Lab 3.2)
# -----------------------------------------------------
# Implements the shared MLM utilities (mask_tokens, build_dataloader, train_bert)
# used by both this script and finetune_lora.py.
#
# When run as __main__:
#   1. Loads bert-base-uncased as BertForMaskedLM
#   2. Fine-tunes on training stories via MLM (3 epochs, AdamW, lr=2e-5)
#   3. Saves checkpoint to results/models/bert_finetuned_checkpoint/
#   4. Extracts word embeddings (mean-pooled subword hidden states)
#   5. Runs preprocess pipeline (downsample → align → trim → lag)
#   6. Saves design matrices for both subjects
#
# Outputs:
#   results/models/bert_finetuned_checkpoint/
#   results/design_matrices/X_bert_finetuned_subj{2,3}_train.npy
#   results/design_matrices/X_bert_finetuned_subj{2,3}_test.npy

import sys
import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths

DATA_DIR  = '/ocean/projects/mth250011p/shared/215a/final_project/data'
_RESULTS  = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUT_DIR   = os.path.join(_RESULTS, 'design_matrices')
MODEL_DIR = os.path.join(_RESULTS, 'models')

SUBJECTS = ['subject2', 'subject3']
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED     = 42


def set_seed(seed=SEED):
    """Set NumPy/PyTorch seeds so fine-tuning and chunk shuffling are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# MLM utilities
# ---------------------------------------------------------------------------

def mask_tokens(
    input_ids,
    vocab_size,
    mask_token_id,
    pad_token_id,
    special_tokens_mask=None,
    mlm_prob=0.15,
):
    """
    Apply MLM masking: 15% of non-padding tokens are selected, then
    80% replaced with [MASK], 10% with a random token, 10% left unchanged.
    Returns masked input_ids and labels (-100 for unmasked positions).
    """
    dev = input_ids.device
    labels = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mlm_prob, device=dev)
    prob_matrix[input_ids == pad_token_id] = 0.0
    if special_tokens_mask is not None:
        prob_matrix[special_tokens_mask.bool()] = 0.0
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=dev)).bool() & masked_indices
    input_ids[replace_mask] = mask_token_id

    # 10% -> random token (50% of the non-[MASK] masked tokens = 10% overall)
    replace_random = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5, device=dev)).bool()
        & masked_indices
        & ~replace_mask
    )
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
    input_ids[replace_random] = random_words[replace_random]

    return input_ids, labels


class _ChunkedTextDataset(Dataset):
    """Small Dataset wrapper for the fixed-length BERT chunks used in MLM training."""

    def __init__(self, input_ids, attention_mask, special_tokens_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.special_tokens_mask = special_tokens_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'special_tokens_mask': self.special_tokens_mask[idx],
        }


def build_dataloader(wordseqs, stories, tokenizer, batch_size=8, max_length=512, stride=256):
    """
    Tokenize training stories into overlapping BERT-length chunks for MLM.

    The caller passes `TRAIN_STORIES` during fine-tuning, so held-out test
    stories are never used to update BERT. Overlapping chunks keep long natural
    stories within BERT's 512-token limit while preserving local context.
    """
    examples = []

    for story in stories:
        words = [str(w) for w in wordseqs[story].data if w]
        text = " ".join(words)
        ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]

        for start in range(0, len(ids), stride):
            chunk = ids[start:start + max_length]
            if len(chunk) >= 32:
                examples.append(chunk)
            if start + max_length >= len(ids):
                break

    enc = tokenizer.pad(
        [{"input_ids": x} for x in examples],
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )

    special = [
        tokenizer.get_special_tokens_mask(x, already_has_special_tokens=True)
        for x in enc["input_ids"].tolist()
    ]
    special = torch.tensor(special, dtype=torch.long)

    dataset = _ChunkedTextDataset(enc["input_ids"], enc["attention_mask"], special)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    """
    Fine-tune a BertForMaskedLM (or PEFT-wrapped variant) with AdamW and MLM loss.
    Prints per-epoch average loss.

    Args:
        model:      BertForMaskedLM or PEFT model with an MLM head
        dataloader: DataLoader from build_dataloader (dict batches)
        tokenizer:  BertTokenizerFast (for vocab/mask/pad token IDs)
        epochs:     number of training epochs
        lr:         AdamW learning rate
        device:     'cuda' or 'cpu'
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)

            input_ids_masked, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                special_tokens_mask=special_tokens_mask,
            )

            outputs = model(
                input_ids=input_ids_masked,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()

        print(f"Epoch {epoch + 1}: loss = {total_loss / len(dataloader):.4f}")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_word_embeddings(wordseqs, stories, model, tokenizer, device, chunk_size=96, stride=48):
    """
    Extract one 768-d embedding per word token from a fine-tuned BertForMaskedLM.

    Runs BERT on overlapping story chunks so each token is represented in
    context, then mean-pools and averages subword states that belong to each
    original word. Empty words get a zero vector.

    This is identical to the strategy in embeddings_bert_pretrained.py so that
    pre-trained vs. fine-tuned results are directly comparable.

    Args:
        wordseqs:  dict of {story: DataSequence} from raw_text.pkl
        stories:   list of story names to process
        model:     BertForMaskedLM (fine-tuned)
        tokenizer: BertTokenizerFast for bert-base-uncased
        device:    'cuda' or 'cpu'

    Returns:
        dict of {story: (n_words, 768) float32 array}
    """
    model.eval()

    word_vectors = {}

    for story in stories:
        words = wordseqs[story].data
        embedding_sums = np.zeros((len(words), 768), dtype=np.float32)
        embedding_counts = np.zeros(len(words), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(words), stride):
                chunk_words = words[start:start + chunk_size]
                valid_positions = [i for i, word in enumerate(chunk_words) if word]
                if not valid_positions:
                    continue

                valid_words = [chunk_words[i] for i in valid_positions]
                inputs = tokenizer(
                    valid_words,
                    is_split_into_words=True,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                )
                word_ids = inputs.word_ids(batch_index=0)
                inputs = inputs.to(device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][0]

                # Fast tokenizer word_ids map subword pieces back to the input
                # word index. Pooling those pieces gives one vector per word.
                for local_idx, original_idx in enumerate(valid_positions):
                    token_positions = [
                        tok_idx for tok_idx, word_id in enumerate(word_ids)
                        if word_id == local_idx
                    ]
                    if token_positions:
                        pooled = hidden[token_positions].mean(dim=0)
                        word_idx = start + original_idx
                        embedding_sums[word_idx] += pooled.cpu().numpy()
                        embedding_counts[word_idx] += 1.0

        nonzero = embedding_counts > 0
        embeddings = np.zeros_like(embedding_sums)
        # A word can appear in more than one overlapping chunk; average all
        # available contextual views to avoid privileging one chunk boundary.
        embeddings[nonzero] = embedding_sums[nonzero] / embedding_counts[nonzero, None]
        word_vectors[story] = embeddings
        print(f"  {story}: {word_vectors[story].shape}")

    return word_vectors


# ---------------------------------------------------------------------------
# Main: fine-tune + extract embeddings
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    set_seed()
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('Loading wordseqs...')
    wordseqs    = pickle.load(open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb'))
    all_stories = TRAIN_STORIES + TEST_STORIES

    print('Loading tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')

    print(f'\nLoading BertForMaskedLM on {DEVICE}...')
    model = BertForMaskedLM.from_pretrained('google-bert/bert-base-uncased').to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'Full fine-tuning trainable parameters: {trainable:,} / {total:,} '
          f'({100 * trainable / total:.2f}%)')

    print('\nBuilding dataloader from training stories only (no leakage)...')
    dataloader = build_dataloader(wordseqs, TRAIN_STORIES, tokenizer)
    print(f'  {len(dataloader.dataset)} chunks, {len(dataloader)} batches per epoch')

    print('\nFine-tuning BERT via MLM...')
    train_bert(model, dataloader, tokenizer, epochs=3, lr=2e-5, device=DEVICE)

    ckpt_path = os.path.join(MODEL_DIR, 'bert_finetuned_checkpoint')
    model.save_pretrained(ckpt_path)
    print(f'\nSaved checkpoint -> {ckpt_path}')

    # Extract embeddings and save design matrices for each subject.
    # Word vectors are the same for both subjects since they heard the same stories;
    # alignment to Y lengths is done per-subject since scan lengths may differ.
    print('\nExtracting fine-tuned BERT embeddings...')
    word_vectors = get_word_embeddings(wordseqs, all_stories, model, tokenizer, DEVICE)

    for subject in SUBJECTS:
        print(f'\n=== {subject} ===')

        print('Downsampling from word-rate to TR-rate via Lanczos filter...')
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print('Aligning X to Y lengths...')
        X_aligned = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print('Trimming edges and applying HRF delays...')
        X_proc = apply_trim_and_lag(X_aligned)

        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test  = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id    = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f'X_bert_finetuned_subj{subj_id}_train.npy')
        test_path  = os.path.join(OUT_DIR, f'X_bert_finetuned_subj{subj_id}_test.npy')

        np.save(train_path, X_train)
        np.save(test_path,  X_test)

        print(f'Saved: {train_path} {X_train.shape}')
        print(f'Saved: {test_path}  {X_test.shape}')

    print('\nDone.')
