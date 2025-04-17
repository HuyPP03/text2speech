import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Tải từ điển
vocab_data = torch.load('translation_vocab.pth')
vi_vocab = vocab_data['vi_vocab']
en_vocab = vocab_data['en_vocab']
idx_to_vi = vocab_data['idx_to_vi']
idx_to_en = vocab_data['idx_to_en']

# Định nghĩa lớp Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Để kết hợp hướng thuận và ngược trong LSTM hai chiều
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        embedding = self.dropout(self.embedding(x))
        # embedding: (batch_size, seq_length, embedding_size)
        
        # outputs: (batch_size, seq_length, hidden_size*2)
        # hidden: (num_layers*2, batch_size, hidden_size)
        # cell: (num_layers*2, batch_size, hidden_size)
        outputs, (hidden, cell) = self.lstm(embedding)
        
        # Tách hướng thuận và ngược
        hidden_forward = hidden[0:self.num_layers]
        hidden_backward = hidden[self.num_layers:]
        cell_forward = cell[0:self.num_layers]
        cell_backward = cell[self.num_layers:]
        
        # Kết hợp
        hidden = torch.cat((hidden_forward, hidden_backward), dim=2)
        cell = torch.cat((cell_forward, cell_backward), dim=2)
        
        # Chuyển về kích thước phù hợp cho decoder
        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        
        return outputs, (hidden, cell)

# Định nghĩa lớp Decoder với cơ chế Attention
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size + hidden_size,  # Kết hợp context vector với embedding
            hidden_size, 
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.energy = nn.Linear(hidden_size * 3, 1)  # 3 = 2 (bidirectional encoder) + 1 (decoder)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_outputs, hidden, cell):
        # x: (batch_size, 1)
        # encoder_outputs: (batch_size, seq_length, hidden_size*2)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedding = self.dropout(self.embedding(x))  # (batch_size, 1, embedding_size)
        
        # Tính attention weights
        batch_size = x.shape[0]
        seq_length = encoder_outputs.shape[1]
        
        # Lặp lại hidden state cho mỗi bước thời gian trong encoder_outputs
        hidden_expanded = hidden[-1].unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_size)
        
        # Tính attention scores
        energy_input = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # (batch_size, seq_length, hidden_size*3)
        energy = self.energy(energy_input).squeeze(2)  # (batch_size, seq_length)
        attention = self.softmax(energy)  # (batch_size, seq_length)
        
        # Tính context vector
        attention = attention.unsqueeze(2)  # (batch_size, seq_length, 1)
        context_vector = torch.bmm(encoder_outputs.transpose(1, 2), attention).squeeze(2)  # (batch_size, hidden_size*2)
        
        # Tinh chỉnh kích thước context vector để phù hợp với hidden_size
        context_vector = context_vector[:, :self.hidden_size]  # (batch_size, hidden_size)
        
        # Kết hợp context vector với embedding
        context_vector = context_vector.unsqueeze(1)  # (batch_size, 1, hidden_size)
        lstm_input = torch.cat((embedding, context_vector), dim=2)  # (batch_size, 1, embedding_size + hidden_size)
        
        # Đưa qua LSTM
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # outputs: (batch_size, 1, hidden_size)
        
        # Dự đoán
        predictions = self.fc(outputs.squeeze(1))  # (batch_size, output_size)
        
        return predictions, hidden, cell

# Mô hình Seq2Seq đầy đủ
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: (batch_size, source_seq_length)
        # target: (batch_size, target_seq_length)
        
        batch_size = source.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Khởi tạo tensor để lưu dự đoán
        outputs = torch.zeros(batch_size, target_length, target_vocab_size).to(self.device)
        
        # Mã hóa nguồn
        encoder_outputs, (hidden, cell) = self.encoder(source)
        
        # Lấy token đầu tiên của target (SOS)
        x = target[:, 0]
        
        # Giải mã từng token
        for t in range(1, target_length):
            # Dự đoán token tiếp theo
            output, hidden, cell = self.decoder(x, encoder_outputs, hidden, cell)
            
            # Lưu dự đoán
            outputs[:, t, :] = output
            
            # Quyết định xem có sử dụng teacher forcing hay không
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Lấy token có xác suất cao nhất
            top1 = output.argmax(1)
            
            # Nếu teacher forcing, sử dụng token thực tế, nếu không sử dụng token dự đoán
            x = target[:, t] if teacher_force else top1
        
        return outputs

# Hàm huấn luyện
def train_lstm_model(train_loader, val_loader, device, save_path='lstm_model.pth'):
    # Tham số mô hình
    input_size_vi = len(vi_vocab)
    input_size_en = len(en_vocab)
    output_size = len(en_vocab)
    embedding_size = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.3
    learning_rate = 0.001
    num_epochs = 5
    
    # Khởi tạo mô hình
    encoder = Encoder(input_size_vi, embedding_size, hidden_size, num_layers, dropout).to(device)
    decoder = Decoder(input_size_en, embedding_size, hidden_size, output_size, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Lấy dữ liệu
            vi_indices = batch['vi_indices'].to(device)
            en_indices = batch['en_indices'].to(device)
            
            # Đặt lại gradient
            optimizer.zero_grad()
            
            # Forward pass
            output = model(vi_indices, en_indices)
            
            # Tính mất mát (bỏ qua SOS token)
            output = output[:, 1:].reshape(-1, output.shape[2])
            target = en_indices[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            
            # Backward pass và cập nhật
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
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
                
                output = model(vi_indices, en_indices, teacher_forcing_ratio=0)
                
                output = output[:, 1:].reshape(-1, output.shape[2])
                target = en_indices[:, 1:].reshape(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'model_params': {
                    'input_size_vi': input_size_vi,
                    'input_size_en': input_size_en,
                    'output_size': output_size,
                    'embedding_size': embedding_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                },
                'vi_vocab': vi_vocab,
                'en_vocab': en_vocab,
                'idx_to_vi': idx_to_vi,
                'idx_to_en': idx_to_en
            }, save_path)
            print(f"Đã lưu mô hình tốt nhất tại {save_path}")
    
    return model

# Hàm dịch câu
def translate_lstm(model, sentence, vi_vocab, en_vocab, idx_to_en, device, max_length=50):
    model.eval()
    
    # Tokenize câu nguồn
    tokens = ['<SOS>'] + sentence.lower().split() + ['<EOS>']
    
    # Chuyển thành indices
    vi_indices = [vi_vocab.get(token, vi_vocab['<UNK>']) for token in tokens]
    
    # Đệm nếu cần
    if len(vi_indices) < max_length:
        vi_indices += [vi_vocab['<PAD>']] * (max_length - len(vi_indices))
    else:
        vi_indices = vi_indices[:max_length]
    
    # Chuyển sang tensor
    vi_tensor = torch.tensor([vi_indices], dtype=torch.long).to(device)
    
    # Mã hóa câu nguồn
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(vi_tensor)
    
    # Bắt đầu với token <SOS>
    x = torch.tensor([en_vocab['<SOS>']], dtype=torch.long).to(device)
    
    # Giải mã từng token
    translated_sentence = []
    
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden, cell = model.decoder(x, encoder_outputs, hidden, cell)
        
        # Lấy token có xác suất cao nhất
        best_guess = output.argmax(1).item()
        
        # Thêm vào câu dịch
        if best_guess == en_vocab['<EOS>']:
            break
        
        if best_guess != en_vocab['<PAD>'] and best_guess != en_vocab['<SOS>']:
            translated_sentence.append(idx_to_en[best_guess])
        
        # Cập nhật đầu vào cho bước tiếp theo
        x = torch.tensor([best_guess], dtype=torch.long).to(device)
    
    return ' '.join(translated_sentence)

# # # Huấn luyện mô hình LSTM (bỏ comment để chạy)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')
# train_lstm_model(train_loader, val_loader, device)

# Hàm để tải và sử dụng mô hình LSTM đã lưu
def load_lstm_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Lấy tham số mô hình
    model_params = checkpoint['model_params']
    
    # Tạo lại mô hình
    encoder = Encoder(
        model_params['input_size_vi'],
        model_params['embedding_size'],
        model_params['hidden_size'],
        model_params['num_layers'],
        model_params['dropout']
    ).to(device)
    
    decoder = Decoder(
        model_params['input_size_en'],
        model_params['embedding_size'],
        model_params['hidden_size'],
        model_params['output_size'],
        model_params['num_layers'],
        model_params['dropout']
    ).to(device)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Tải trọng số
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Tải từ điển
    vi_vocab = checkpoint['vi_vocab']
    en_vocab = checkpoint['en_vocab']
    idx_to_vi = checkpoint['idx_to_vi']
    idx_to_en = checkpoint['idx_to_en']
    
    return model, vi_vocab, en_vocab, idx_to_vi, idx_to_en

