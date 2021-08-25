import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import cog

model_path = (
    "models/RRDB_ESRGAN_x4.pth"
)


class Predictor(cog.Predictor):
    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("Loading model...")
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    @cog.input("image", type=Path, help="Low-resolution input image")
    def predict(self, image):
        print("Reading input image...")
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        print("Upscaling...")
        with torch.no_grad():
            output = (
                self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        out_path = Path(tempfile.mkdtemp()) / "out.png"

        print("Saving result...")
        cv2.imwrite(str(out_path), output)

        return out_path
