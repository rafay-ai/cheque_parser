"""
model_date.py
DateNet CRNN architecture + DateReader inference class.
Reads handwritten dates in DD/MM/YYYY format from a cropped image region.
"""

import re

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class DateNet(nn.Module):
    """
    CRNN: CNN backbone → BiLSTM → CTC decode
    Input:  (B, 1, 32, 256) grayscale
    Output: (B, W', 12)   where 12 = blank + digits 0-9 + '/'
    """

    def __init__(self, rnn_input_size: int = 256):
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
            nn.MaxPool2d((2, 1), (2, 1)),  # 11 height-only pool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 12
            nn.BatchNorm2d(256),  # 13
            nn.ReLU(inplace=True),  # 14
        )
        # Checkpoint shows weight_ih_l0 shape (1024, 256) → input_size=256
        # meaning height collapses to 1 after pooling (256 channels * 1 = 256)
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # 256*2 bidirectional = 512
        self.fc = nn.Linear(512, 12)

    def forward(self, x):
        x = self.cnn(x)  # (B, 256, H', W')
        b, c, h, w = x.size()
        # Average pool over height to reduce to 256
        x = x.mean(dim=2)  # (B, 256, W')
        x = x.permute(0, 2, 1)  # (B, W', 256)
        x, _ = self.rnn(x)  # (B, W', 512)
        x = self.fc(x)  # (B, W', 12)
        return x


class DateReader:
    """
    Loads date_model.pt and runs CTC-greedy inference on a crop.
    Usage:
        reader = DateReader("models/date_model.pt")
        result = reader.predict(pil_image_or_path)
        # result = {"date": "12/03/2026", "confidence": 0.87, "valid": True}
    """

    # CTC vocab: 0=blank, 1-10 → digits '0'-'9', 11 → '/'
    CHAR_MAP = {i: str(i - 1) for i in range(1, 11)}
    CHAR_MAP[11] = "/"
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
            # Auto-detect input_size from the checkpoint weights
            # weight_ih_l0 shape is (4*hidden, input_size)
            input_size = state["rnn.weight_ih_l0"].shape[1]
            print(f"  [DateReader] detected rnn input_size={input_size}")
            self.model = DateNet(rnn_input_size=input_size).to(self.device)
            self.model.load_state_dict(state)

        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        print(f"[DateReader] loaded on {self.device}")

    def predict(self, image, min_conf: float = 0.4) -> dict:
        """
        Args:
            image: PIL.Image or file path string
            min_conf: minimum average confidence to mark result as valid
        Returns:
            {"date": "DD/MM/YYYY" or None, "confidence": float, "valid": bool}
        """
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)  # (1, W', 12)
            probs = torch.softmax(output, dim=2)
            pred = probs.argmax(dim=2)[0]  # (W',)
            confs = probs.max(dim=2).values[0]

        # CTC greedy decode (collapse repeats, remove blanks)
        decoded, conf_vals, prev = [], [], None
        for i, p in enumerate(pred.tolist()):
            if p != prev and p != self.BLANK:
                decoded.append(p)
                conf_vals.append(confs[i].item())
            prev = p

        raw_str = "".join(self.CHAR_MAP.get(d, "") for d in decoded)
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
        print(f"  [DateReader] raw='{raw_str}'  conf={avg_conf:.3f}")

        date_str = self._parse_date(raw_str)
        valid = date_str is not None and avg_conf >= min_conf
        return {"date": date_str, "confidence": round(avg_conf, 4), "valid": valid}

    @staticmethod
    def _parse_date(s: str):
        # Try with separators already present
        m = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", s)
        if m:
            dd, mm, yy = m.group(1), m.group(2), m.group(3)
            yy = "20" + yy if len(yy) == 2 else yy
            if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
                return f"{dd.zfill(2)}/{mm.zfill(2)}/{yy}"

        # Try as plain 8-digit string
        digits = re.sub(r"\D", "", s)
        for candidate in [digits, digits[-8:] if len(digits) >= 8 else ""]:
            if len(candidate) == 8:
                try:
                    dd, mm, yy = (
                        int(candidate[:2]),
                        int(candidate[2:4]),
                        int(candidate[4:]),
                    )
                    if 1 <= dd <= 31 and 1 <= mm <= 12 and 2000 <= yy <= 2099:
                        return f"{candidate[:2]}/{candidate[2:4]}/{candidate[4:]}"
                except ValueError:
                    pass
        return None
