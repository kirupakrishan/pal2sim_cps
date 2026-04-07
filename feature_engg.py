import torch
import torch.nn.functional as F
import numpy as np

class FeatureEngineer:
    def __init__(self,Seed=42):
        np.random.seed(Seed)
        torch.manual_seed(Seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    # =========================================================
    # HELPERS
    # =========================================================
    def print_shapes(self,name, X, y=None):
        if y is None:
            print(f"{name}: {X.shape}, dtype={X.dtype}")
        else:
            print(f"{name}: X={X.shape}, y={y.shape}, X_dtype={X.dtype}, y_dtype={y.dtype}")
            
    # =========================================================
    # 1. REDUCE RAW SIGNAL TO (N, 7, 16)
    # =========================================================
    def reduce_raw_to_16(self,X, out_points=16, chunk_size=10000):
        chunks = []

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])

            chunk = torch.from_numpy(X[start:end]).to(device=self.device, dtype=torch.float32)
            chunk_reduced = F.adaptive_avg_pool1d(chunk, out_points)   # (B, 7, 16)

            chunks.append(chunk_reduced.cpu().numpy())

            print(f"Raw chunk {start}:{end} -> {tuple(chunk_reduced.shape)}")

            del chunk, chunk_reduced
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(chunks, axis=0).astype(np.float32)

    # =========================================================
    # 2. REDUCE FFT TO (N, 7, 16)
    # =========================================================
    def compute_reduced_rfft_features(self,X, out_bins=16, chunk_size=10000, use_log=True):
        fft_chunks = []

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])

            chunk = torch.from_numpy(X[start:end]).to(device=self.device, dtype=torch.float32)

            chunk_fft = torch.fft.rfft(chunk, dim=-1)   # (B, 7, 81)
            chunk_fft = torch.abs(chunk_fft).float()

            if use_log:
                chunk_fft = torch.log1p(chunk_fft)

            chunk_fft_reduced = F.adaptive_avg_pool1d(chunk_fft, out_bins)   # (B, 7, 16)

            fft_chunks.append(chunk_fft_reduced.cpu().numpy())

            print(f"FFT chunk {start}:{end} -> {tuple(chunk_fft_reduced.shape)}")

            del chunk, chunk_fft, chunk_fft_reduced
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(fft_chunks, axis=0).astype(np.float32)

    # =========================================================
    # 3. ENTROPY OF 16 LOCAL CHUNKS -> (N, 7, 16)
    # =========================================================
    def compute_entropy_16_features(self,X, bins=16, chunk_size=10000):
        """
        Input:
            X shape = (N, 7, 160)

        Output:
            entropy_features shape = (N, 7, 16)

        Method:
            Split each 160-sample window into 16 chunks of length 10.
            Compute histogram entropy for each chunk.
        """
        assert X.shape[2] == 160, "Expected last dimension = 160"
        assert 160 % 16 == 0, "Window length must be divisible by 16"

        sub_len = 160 // 16   # 10
        entropy_chunks = []

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])

            chunk = torch.from_numpy(X[start:end]).to(device=self.device, dtype=torch.float32)  # (B, 7, 160)
            B, C, T = chunk.shape

            # reshape into 16 local segments of length 10
            chunk_reshaped = chunk.view(B, C, 16, sub_len)   # (B, 7, 16, 10)

            x_min = chunk_reshaped.min(dim=-1, keepdim=True).values
            x_max = chunk_reshaped.max(dim=-1, keepdim=True).values
            x_range = x_max - x_min

            valid = x_range > 0

            # normalize to [0,1]
            scaled = torch.where(valid, (chunk_reshaped - x_min) / x_range, torch.zeros_like(chunk_reshaped))

            # bin indices
            bin_idx = torch.floor(scaled * bins).long()
            bin_idx = torch.clamp(bin_idx, 0, bins - 1)   # (B, 7, 16, 10)

            # histogram counts
            counts = torch.zeros((B, C, 16, bins), device=self.device, dtype=torch.float32)
            counts.scatter_add_(
                dim=-1,
                index=bin_idx,
                src=torch.ones_like(bin_idx, dtype=torch.float32)
            )  # (B, 7, 16, bins)

            probs = counts / sub_len

            entropy = -torch.sum(
                torch.where(probs > 0, probs * torch.log2(probs), torch.zeros_like(probs)),
                dim=-1
            )  # (B, 7, 16)

            entropy = torch.where(valid.squeeze(-1), entropy, torch.zeros_like(entropy))

            entropy_chunks.append(entropy.cpu().numpy())

            print(f"Entropy chunk {start}:{end} -> {tuple(entropy.shape)}")

            del chunk, chunk_reshaped, x_min, x_max, x_range, valid, scaled, bin_idx, counts, probs, entropy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(entropy_chunks, axis=0).astype(np.float32)


    # =========================================================
    # 4. BUILD FINAL 21x16 FEATURES
    # =========================================================
    def build_21x16_features(self,X, chunk_size=10000, fft_log=True, entropy_bins=16):
        """
        Input:
            X shape = (N, 7, 160)

        Output:
            X_21x16 shape = (N, 21, 16)
        """
        raw_16 = self.reduce_raw_to_16(X, out_points=16, chunk_size=chunk_size)  # (N, 7, 16)
        fft_16 = self.compute_reduced_rfft_features(X, out_bins=16, chunk_size=chunk_size, use_log=fft_log)  # (N, 7, 16)
        ent_16 = self.compute_entropy_16_features(X, bins=entropy_bins, chunk_size=chunk_size)  # (N, 7, 16)

        X_21x16 = np.concatenate([raw_16, fft_16, ent_16], axis=1).astype(np.float32)
        return X_21x16

    def flatten_features(self,X):
        """
        Flatten 3D tensor into 2D for classical ML.

        Input:
            (N, 14, 160)s

        Output:
            (N, 14*160)
        """
        return X.reshape(X.shape[0], -1).astype(np.float16)



