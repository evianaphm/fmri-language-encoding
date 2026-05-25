# code/lab32/finetune_lora.py
#
# LoRA fine-tuning of BERT for fMRI encoding (Lab 3.2)
# -----------------------------------------------------
# Wraps bert-base-uncased with Low-Rank Adaptation (LoRA) adapters and
# fine-tunes on training stories via masked language modeling (MLM).
# LoRA freezes most of BERT's weights and only trains small low-rank
# matrices injected into the attention layers, dramatically reducing
# the number of trainable parameters vs full fine-tuning.
#
# After fine-tuning, extracts one 768-d embedding per word token using
# mean pooling over subword hidden states, then passes through the
# standard preprocess.py pipeline (downsample -> trim -> lag) to
# produce design matrices for ridge regression.
#
# Outputs:
#   results/design_matrices/X_bert_lora_subj{2,3}_train.npy
#   results/design_matrices/X_bert_lora_subj{2,3}_test.npy
#   results/models/bert_lora_checkpoint/

import sys
import os
import numpy as np
import pickle
import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from peft import LoraConfig, get_peft_model

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
    """Set NumPy/PyTorch seeds so LoRA fine-tuning is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# LoRA hyperparameters
# r: rank of the low-rank matrices — lower = fewer params, higher = more expressive
# lora_alpha: scaling factor for LoRA updates (effective lr scales as alpha/r)
# target_modules: which attention weight matrices to inject LoRA into
# lora_dropout: regularization on the LoRA layers
LORA_R       = 8
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_MODULES = ["query", "value"]


def build_lora_model():
    """
    Load bert-base-uncased and wrap it with LoRA adapters for MLM.

    LoRA injects trainable low-rank matrices (A, B) into the query and value
    projection layers of every attention head. The original weights are frozen;
    only A and B are updated during training. This reduces trainable parameters
    from ~110M (full fine-tuning) to ~1-2M while still adapting the model.

    Returns:
        PeftModel wrapping BertForMaskedLM with LoRA adapters
    """
    base_model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    # Report and compare parameter counts — key result for check-in
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.4f}%)")

    return model


def get_word_embeddings(wordseqs, stories, model, tokenizer, device, chunk_size=96, stride=48):
    """
    Extract one 768-d embedding per word token from the LoRA fine-tuned model.

    Runs BERT on overlapping story chunks so each token is represented in
    context, then mean-pools and averages subword states that belong to each
    original word. Empty words get a zero vector.

    This is identical to the strategy in embeddings_bert_pretrained.py so that
    pre-trained vs LoRA fine-tuned results are directly comparable.

    Args:
        wordseqs: dict of {story: DataSequence} from raw_text.pkl
        stories:  list of story names to process
        model:    LoRA fine-tuned PeftModel (BertForMaskedLM wrapper)
        tokenizer: BertTokenizerFast for bert-base-uncased
        device:   'cuda' or 'cpu'

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

                outputs = model(output_hidden_states=True, **inputs)
                hidden = outputs.hidden_states[-1][0]

                # Map BERT subword pieces back to original story words, then
                # mean-pool so every downstream row still corresponds to a word.
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
        # Overlapping windows give some words multiple contextual embeddings.
        # Averaging keeps the representation independent of arbitrary chunk cuts.
        embeddings[nonzero] = embedding_sums[nonzero] / embedding_counts[nonzero, None]
        word_vectors[story] = embeddings
        print(f"  {story}: {word_vectors[story].shape}")

    return word_vectors


if __name__ == "__main__":
    set_seed()
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading wordseqs...")
    wordseqs    = pickle.load(open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb'))
    all_stories = TRAIN_STORIES + TEST_STORIES

    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    # Build LoRA model — prints trainable parameter count for check-in report
    print(f"\nBuilding LoRA model on {DEVICE}...")
    lora_model = build_lora_model().to(DEVICE)

    # Fine-tune using MLM training loop and dataloader
    from finetune_bert import train_bert, build_dataloader

    print("\nBuilding dataloader from training stories only (no leakage)...")
    dataloader = build_dataloader(wordseqs, TRAIN_STORIES, tokenizer)

    print("\nFine-tuning LoRA model via MLM...")
    train_bert(lora_model, dataloader, tokenizer, epochs=3, lr=5e-4, device=DEVICE)

    # Save checkpoint so embeddings can be re-extracted without retraining
    ckpt_path = os.path.join(MODEL_DIR, 'bert_lora_checkpoint')
    lora_model.save_pretrained(ckpt_path)
    print(f"\nSaved LoRA checkpoint -> {ckpt_path}")

    # Extract embeddings once; only alignment to Y is subject-specific.
    print("\nExtracting LoRA BERT embeddings...")
    word_vectors = get_word_embeddings(
        wordseqs, all_stories, lora_model, tokenizer, DEVICE
    )

    # Save design matrices for each subject.
    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")

        print("Downsampling from word-rate to TR-rate via Lanczos filter...")
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print("Aligning X to Y lengths...")
        X_aligned = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print("Trimming edges and applying HRF delays...")
        X_proc = apply_trim_and_lag(X_aligned)

        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test  = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id    = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f'X_bert_lora_subj{subj_id}_train.npy')
        test_path  = os.path.join(OUT_DIR, f'X_bert_lora_subj{subj_id}_test.npy')

        np.save(train_path, X_train)
        np.save(test_path,  X_test)

        print(f"Saved: {train_path} {X_train.shape}")
        print(f"Saved: {test_path}  {X_test.shape}")

    print("\nDone.")
