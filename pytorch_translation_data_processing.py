import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Tải dataset từ Hugging Face
dataset = load_dataset("ILT37/translate_vi_en")


# Lớp dataset tùy chỉnh
class TranslationDataset(Dataset):
    def __init__(self, data, vi_vocab, en_vocab, max_length=30):
        self.data = data
        self.vi_vocab = vi_vocab
        self.en_vocab = en_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        vi_text = item['vi']
        en_text = item['en']
        
        # Tokenize và chuyển sang indices
        vi_tokens = ['<SOS>'] + [token for token in vi_text.lower().split()] + ['<EOS>']
        en_tokens = ['<SOS>'] + [token for token in en_text.lower().split()] + ['<EOS>']
        
        # Cắt hoặc đệm các câu
        vi_indices = self.tokens_to_indices(vi_tokens, self.vi_vocab)
        en_indices = self.tokens_to_indices(en_tokens, self.en_vocab)
        
        # Chuyển sang tensor
        return {
            'vi_text': vi_text,
            'en_text': en_text,
            'vi_indices': torch.tensor(vi_indices, dtype=torch.long),
            'en_indices': torch.tensor(en_indices, dtype=torch.long),
            'vi_length': len(vi_tokens),
            'en_length': len(en_tokens)
        }
    
    def tokens_to_indices(self, tokens, vocab):
        indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Cắt nếu dài quá
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        
        # Đệm nếu ngắn
        padding = [vocab['<PAD>']] * (self.max_length - len(indices))
        indices = indices + padding
        
        return indices

# Hàm xây dựng từ điển
def build_vocab(dataset, min_freq=5):
    word_counts = {}
    
    # Đếm tần suất từ
    for item in tqdm(dataset, desc="Counting words"):
        vi_tokens = item['vi'].lower().split()
        en_tokens = item['en'].lower().split()
        
        for token in vi_tokens:
            word_counts[('vi', token)] = word_counts.get(('vi', token), 0) + 1
        
        for token in en_tokens:
            word_counts[('en', token)] = word_counts.get(('en', token), 0) + 1
    
    # Xây dựng từ điển
    vi_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    en_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    vi_idx = 4
    en_idx = 4
    
    for (lang, token), count in word_counts.items():
        if count >= min_freq:
            if lang == 'vi' and token not in vi_vocab:
                vi_vocab[token] = vi_idx
                vi_idx += 1
            elif lang == 'en' and token not in en_vocab:
                en_vocab[token] = en_idx
                en_idx += 1
    
    return vi_vocab, en_vocab

# Xây dựng từ điển từ tập huấn luyện
vi_vocab, en_vocab = build_vocab(dataset['train'])

# In kích thước từ điển
print(f"Kích thước từ điển tiếng Việt: {len(vi_vocab)}")
print(f"Kích thước từ điển tiếng Anh: {len(en_vocab)}")

# Tạo từ điển ngược
idx_to_vi = {idx: word for word, idx in vi_vocab.items()}
idx_to_en = {idx: word for word, idx in en_vocab.items()}

# Tạo dataset
train_dataset = TranslationDataset(dataset['train'], vi_vocab, en_vocab)
val_dataset = TranslationDataset(dataset['validation'], vi_vocab, en_vocab)
test_dataset = TranslationDataset(dataset['test'], vi_vocab, en_vocab)

# Tạo DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

# # Lưu từ điển để sử dụng sau này
torch.save({
    'vi_vocab': vi_vocab,
    'en_vocab': en_vocab,
    'idx_to_vi': idx_to_vi,
    'idx_to_en': idx_to_en
}, 'translation_vocab.pth')

print("Đã lưu từ điển vào translation_vocab.pth")