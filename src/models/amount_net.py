"""
model_amount.py
AmountNet CRNN architecture + AmountReader inference class.
The amount_model.pt uses the same CRNN structure as the date model
(CNN backbone → BiLSTM → CTC), auto-detected from checkpoint keys.
"""

import re

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ─── Architecture (matches checkpoint keys: cnn.*, rnn.*, fc.*) ──────────────


class AmountNet(nn.Module):
    """
    CRNN: CNN backbone → BiLSTM → CTC decode
    Same structure as DateNet but output vocab size is auto-detected.
    Input:  (B, 1, 32, 256) grayscale
    Output: (B, W', num_classes)
    """

    def __init__(
        self, rnn_input_size: int = 256, hidden_size: int = 256, num_classes: int = 11
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 0
            nn.BatchNorm2d(32),  # 1
            nn.ReLU(inplace=True),  # 2
            nn.MaxPool2d(2, 2),  # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4
            nn.BatchNorm2d(64),  # 5
            nn.ReLU(inplace=True),  # 6
            nn.MaxPool2d(2, 2),  # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8
            nn.BatchNorm2d(128),  # 9
            nn.ReLU(inplace=True),  # 10
            nn.MaxPool2d(2, 2),  # 11 — full 2x2 pool (no height-only)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 12
            nn.BatchNorm2d(256),  # 13
            nn.ReLU(inplace=True),  # 14
        )
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, 256, H', W')
        b, c, h, w = x.size()
        # Collapse height dimension so input to RNN = c * 1 = 256
        x = x.mean(dim=2)  # (B, 256, W')  — avg pool over height
        x = x.permute(0, 2, 1)  # (B, W', 256)
        x, _ = self.rnn(x)  # (B, W', hidden*2)
        x = self.fc(x)  # (B, W', num_classes)
        return x


# ─── AmountReader ─────────────────────────────────────────────────────────────


class AmountReader:
    """
    Loads amount_model.pt (CRNN) and runs CTC-greedy inference on a crop.

    Usage:
        reader = AmountReader("models/amount_model.pt")
        result = reader.predict(pil_image_or_bgr_array)
        # result = {"amount": "34,500", "raw": "34500", "valid": True}
    """

    BLANK = 0

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
        else:
            state = checkpoint.get("state_dict", checkpoint)

            # Auto-detect architecture from checkpoint shapes
            rnn_input_size = state["rnn.weight_ih_l0"].shape[1]
            hidden_size = state["rnn.weight_hh_l0"].shape[1]
            num_classes = state["fc.weight"].shape[0]
            print(
                f"  [AmountReader] rnn_input={rnn_input_size}  hidden={hidden_size}  classes={num_classes}"
            )

            self.model = AmountNet(
                rnn_input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
            ).to(self.device)
            self.model.load_state_dict(state)

        self.model.eval()

        # 0=blank, 1-10 → '0'-'9', 11 → '/'
        self.char_map = {i: str(i - 1) for i in range(1, 11)}
        self.char_map[11] = "/"

        # Use actual rnn_input_size from checkpoint — height is collapsed via mean pooling
        actual_rnn_input = state["rnn.weight_ih_l0"].shape[1]
        print(f"  [AmountReader] actual rnn_input={actual_rnn_input}")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        print(f"[AmountReader] loaded on {self.device}")

    def predict(self, image, min_conf: float = 0.4) -> dict:
        """
        Args:
            image: PIL.Image, BGR numpy array (OpenCV), or file path string
            min_conf: minimum avg confidence to mark result valid
        Returns:
            {"amount": "34,500", "raw": "34500", "valid": bool, "confidence": float}
        """
        if isinstance(image, np.ndarray):
            import cv2

            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)  # (1, W', num_classes)
            probs = torch.softmax(output, dim=2)
            pred = probs.argmax(dim=2)[0]  # (W',)
            confs = probs.max(dim=2).values[0]

        # CTC greedy decode
        decoded, conf_vals, prev = [], [], None
        for i, p in enumerate(pred.tolist()):
            if p != prev and p != self.BLANK:
                decoded.append(p)
                conf_vals.append(confs[i].item())
            prev = p

        raw_str = "".join(self.char_map.get(d, "") for d in decoded)
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
        print(f"  [AmountReader] raw='{raw_str}'  conf={avg_conf:.3f}")

        amount_str, digits_only = self._parse_amount(raw_str)
        valid = amount_str is not None and avg_conf >= min_conf

        return {
            "amount": amount_str,
            "raw": digits_only,
            "confidence": round(avg_conf, 4),
            "valid": valid,
        }

    @staticmethod
    def _parse_amount(s: str):
        digits = re.sub(r"\D", "", s)
        if not digits:
            return None, digits
        if len(set(digits)) == 1:
            print(f"  [AmountReader] all-same digits '{digits}' → noise, skipping")
            return None, digits
        if len(digits) > 8:
            print(f"  [AmountReader] {len(digits)} digits too long → skipping")
            return None, digits
        try:
            val = int(digits)
            if 100 <= val <= 100_000_000:
                formatted = f"{val:,}"
                print(f"  [AmountReader] parsed amount: {formatted}")
                return formatted, digits
        except ValueError:
            pass
        return None, digits
