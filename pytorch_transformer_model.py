import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

# Tải từ điển
vocab_data = torch.load('translation_vocab.pth')
vi_vocab = vocab_data['vi_vocab']
en_vocab = vocab_data['en_vocab']
idx_to_vi = vocab_data['idx_to_vi']
idx_to_en = vocab_data['idx_to_en']

# Lớp Positional Encoding cho Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Mô hình Transformer
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout, max_len, device):
        super(TransformerModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.device = device
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Khởi tạo trọng số
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # src, tgt: (batch_size, seq_len)
        
        # Embedding và positional encoding
        src_embedded = self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model))
        
        # Tạo mask cho target để ngăn chặn attention đến các token tương lai
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        
        # Forward qua transformer
        output = self.transformer(
            src_embedded, tgt_embedded, 
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Linear projection để lấy logits
        output = self.output_layer(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, seq, pad_idx):
        return (seq == pad_idx)

# Hàm huấn luyện Transformer
def train_transformer_model(train_loader, val_loader, device, save_path='transformer_model.pth'):
    # Tham số mô hình
    src_vocab_size = len(vi_vocab)
    tgt_vocab_size = len(en_vocab)
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    max_len = 100
    learning_rate = 0.0001
    num_epochs = 5
    
    # Khởi tạo mô hình
    model = TransformerModel(
        src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers,
        num_decoder_layers, dim_feedforward, dropout, max_len, device
    ).to(device)
    
    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Lấy dữ liệu
            vi_indices = batch['vi_indices'].to(device)
            en_indices = batch['en_indices'].to(device)
            
            # Tạo padding masks
            src_padding_mask = model.create_padding_mask(vi_indices, vi_vocab['<PAD>']).to(device)
            tgt_padding_mask = model.create_padding_mask(en_indices, en_vocab['<PAD>']).to(device)
            
            # Tạo tgt_input (bỏ token cuối cùng) và tgt_output (bỏ token đầu tiên)
            tgt_input = en_indices[:, :-1]
            tgt_output = en_indices[:, 1:]
            
            # Cập nhật padding mask cho target input
            tgt_input_padding_mask = model.create_padding_mask(tgt_input, en_vocab['<PAD>']).to(device)
            
            # Tạo mask để tránh attention đến các token tương lai
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Đặt lại gradient
            optimizer.zero_grad()
            
            # Forward pass
            output = model(
                vi_indices, tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_input_padding_mask
            )
            
            # Tính mất mát
            output = output.reshape(-1, output.shape[2])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            
            # Backward pass và cập nhật
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # In tiến trình sau mỗi 100 batch
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                vi_indices = batch['vi_indices'].to(device)
                en_indices = batch['en_indices'].to(device)
                
                # Tạo padding masks
                src_padding_mask = model.create_padding_mask(vi_indices, vi_vocab['<PAD>']).to(device)
                
                # Tạo tgt_input và tgt_output
                tgt_input = en_indices[:, :-1]
                tgt_output = en_indices[:, 1:]
                
                # Cập nhật padding mask cho target input
                tgt_input_padding_mask = model.create_padding_mask(tgt_input, en_vocab['<PAD>']).to(device)
                
                # Tạo mask
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                # Forward pass
                output = model(
                    vi_indices, tgt_input,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_input_padding_mask
                )
                
                # Tính mất mát
                output = output.reshape(-1, output.shape[2])
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(output, tgt_output)
                val_loss += loss.item()
        
        # Cập nhật learning rate
        scheduler.step()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_params': {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_encoder_layers': num_encoder_layers,
                    'num_decoder_layers': num_decoder_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'max_len': max_len
                },
                'vi_vocab': vi_vocab,
                'en_vocab': en_vocab,
                'idx_to_vi': idx_to_vi,
                'idx_to_en': idx_to_en
            }, save_path)
            print(f"Đã lưu mô hình tốt nhất tại {save_path}")
    
    return model

# Hàm dịch câu với Transformer
def translate_transformer(model, sentence, vi_vocab, en_vocab, idx_to_en, device, max_length=50):
    model.eval()
    
    # Tokenize câu nguồn
    tokens = ['< SOS >'] + sentence.lower().split() + ['<EOS>']
    
    # Chuyển thành indices
    vi_indices = [vi_vocab.get(token, vi_vocab['<UNK>']) for token in tokens]
    
    # Đệm nếu cần
    if len(vi_indices) < max_length:
        vi_indices += [vi_vocab['<PAD>']] * (max_length - len(vi_indices))
    else:
        vi_indices = vi_indices[:max_length]
    
    # Chuyển sang tensor
    src_tensor = torch.tensor([vi_indices], dtype=torch.long).to(device)
    
    # Tạo padding mask cho source
    src_padding_mask = model.create_padding_mask(src_tensor, vi_vocab['<PAD>']).to(device)
    
    # Bắt đầu với token < SOS >
    tgt_tensor = torch.tensor([[en_vocab['< SOS >']]], dtype=torch.long).to(device)
    
    # Dịch từng token
    for i in range(max_length):
        # Tạo mask cho target
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
        
        # Tạo padding mask cho target
        tgt_padding_mask = model.create_padding_mask(tgt_tensor, en_vocab['<PAD>']).to(device)
        
        # Dự đoán
        with torch.no_grad():
            output = model(
                src_tensor, tgt_tensor,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
        
        # Lấy token cuối cùng được dự đoán
        pred_token = output[:, -1, :].argmax(dim=1).unsqueeze(1)
        
        # Thêm vào chuỗi target
        tgt_tensor = torch.cat([tgt_tensor, pred_token], dim=1)
        
        # Nếu là token <EOS>, dừng lại
        if pred_token.item() == en_vocab['<EOS>']:
            break
    
    # Chuyển từ indices sang từ
    predicted_indices = tgt_tensor.squeeze().tolist()
    
    # Bỏ qua < SOS > và <EOS>
    predicted_tokens = []
    for idx in predicted_indices:
        if idx == en_vocab['< SOS >'] or idx == en_vocab['<EOS>'] or idx == en_vocab['<PAD>']:
            continue
        predicted_tokens.append(idx_to_en[idx])
    
    return ' '.join(predicted_tokens)

# Huấn luyện mô hình Transformer (bỏ comment để chạy)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_transformer_model(train_loader, val_loader, device)

# Hàm để tải và sử dụng mô hình Transformer đã lưu
def load_transformer_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Lấy tham số mô hình
    model_params = checkpoint['model_params']
    
    # Tạo lại mô hình
    model = TransformerModel(
        model_params['src_vocab_size'],
        model_params['tgt_vocab_size'],
        model_params['d_model'],
        model_params['nhead'],
        model_params['num_encoder_layers'],
        model_params['num_decoder_layers'],
        model_params['dim_feedforward'],
        model_params['dropout'],
        model_params['max_len'],
        device
    ).to(device)
    
    # Tải trọng số
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Tải từ điển
    vi_vocab = checkpoint['vi_vocab']
    en_vocab = checkpoint['en_vocab']
    idx_to_vi = checkpoint['idx_to_vi']
    idx_to_en = checkpoint['idx_to_en']
    
    return model, vi_vocab, en_vocab, idx_to_vi, idx_to_en