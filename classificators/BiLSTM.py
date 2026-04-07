import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pal2sim_cps.feature_engg import FeatureEngineer


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, seq_len, input_size)

        _, (h_n, _) = self.lstm(x)

        # final hidden state from last BiLSTM layer
        forward_hidden = h_n[-2]   # (B, hidden_size)
        backward_hidden = h_n[-1]  # (B, hidden_size)

        h = torch.cat([forward_hidden, backward_hidden], dim=1)  # (B, 2*hidden_size)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        logits = self.fc2(h)  # (B, num_classes)

        return logits


class BiLSTMClassifier:
    def __init__(
        self,
        target_vals,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        batch_size=64,
        epochs=20,
        threshold=0.5,
        seed=42,
        chunk_size=10000,
        fft_log=True,
        entropy_bins=16
    ):
        self.target_vals = target_vals
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.seed = seed
        self.chunk_size = chunk_size
        self.fft_log = fft_log
        self.entropy_bins = entropy_bins

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer(Seed=seed)

        self._set_seed()

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _to_numpy(self, X, y=None):
        if hasattr(X, "values"):
            X = X.values
        if y is not None and hasattr(y, "values"):
            y = y.values
        return X, y

    def _ensure_raw_shape(self, X):
        """
        Expected raw input before feature engineering: (N, 7, 160)
        """
        if len(X.shape) != 3:
            raise ValueError(f"Expected raw X with shape (N, 7, 160), got {X.shape}")

        if X.shape[1] != 7 or X.shape[2] != 160:
            raise ValueError(f"Expected raw X with shape (N, 7, 160), got {X.shape}")

        return X.astype(np.float32)

    def _build_features(self, X):
        """
        Raw X: (N, 7, 160)
        Feature-engineered X: (N, 21, 16)
        BiLSTM input needed: (N, 16, 21)
        """
        X = self._ensure_raw_shape(X)

        X_feat = self.feature_engineer.build_21x16_features(
            X,
            chunk_size=self.chunk_size,
            fft_log=self.fft_log,
            entropy_bins=self.entropy_bins
        )  # (N, 21, 16)

        X_feat = np.transpose(X_feat, (0, 2, 1))  # -> (N, 16, 21)
        return X_feat.astype(np.float32)

    # def _scale_features(self, X_train, X_val):
    #     """
    #     Scale over feature dimension after reshaping to 2D.
    #     X_train, X_val shapes: (N, seq_len, input_size)
    #     """
    #     n_train, seq_len, input_size = X_train.shape
    #     n_val = X_val.shape[0]

    #     self.scaler = StandardScaler()

    #     X_train_2d = X_train.reshape(-1, input_size)
    #     X_val_2d = X_val.reshape(-1, input_size)

    #     X_train_scaled = self.scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, input_size)
    #     X_val_scaled = self.scaler.transform(X_val_2d).reshape(n_val, seq_len, input_size)

    #     return X_train_scaled.astype(np.float32), X_val_scaled.astype(np.float32)

    # def _scale_test(self, X_test):
    #     n_test, seq_len, input_size = X_test.shape
    #     X_test_2d = X_test.reshape(-1, input_size)
    #     X_test_scaled = self.scaler.transform(X_test_2d).reshape(n_test, seq_len, input_size)
    #     return X_test_scaled.astype(np.float32)

    def train(self, train, val):
        X_train, y_train = train
        X_val, y_val = val

        X_train, y_train = self._to_numpy(X_train, y_train)
        X_val, y_val = self._to_numpy(X_val, y_val)

        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        print("Raw train shape:", X_train.shape)
        print("Raw val shape:", X_val.shape)

        X_train = self._build_features(X_train)   # (N, 16, 21)
        X_val = self._build_features(X_val)       # (N, 16, 21)

        print("Feature train shape:", X_train.shape)
        print("Feature val shape:", X_val.shape)


        input_size = X_train.shape[2]      # 21
        num_classes = y_train.shape[1]     # number of labels

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = BiLSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X):
        X, _ = self._to_numpy(X)
        X = self._build_features(X)     # (N, 16, 21)

        dummy_y = np.zeros((len(X), len(self.target_vals)), dtype=np.float32)
        dataset = TimeSeriesDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        probs_all = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits)
                probs_all.append(probs.cpu().numpy())

        probs_all = np.vstack(probs_all)
        preds = (probs_all >= self.threshold).astype(np.int32)
        return preds