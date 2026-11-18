"""Entrena la CNN ligera. Estructura esperada:
dataset/mancha/*.jpg   (recortes con mancha)
dataset/ok/*.jpg       (recortes sin mancha: sombras, estampados, costuras, tela limpia)
Uso: python train_cnn.py"""
import glob, cv2, torch, torch.nn as nn
from src.cnn import build_net, to_tensor

X, y = [], []
for label, folder in ((1.0, "mancha"), (0.0, "ok")):
    for f in glob.glob(f"dataset/{folder}/*"):
        img = cv2.imread(f)
        if img is not None:
            X.append(to_tensor(img)); y.append(label)
X = torch.stack(X); y = torch.tensor(y).unsqueeze(1)
net = build_net(); opt = torch.optim.Adam(net.parameters(), 1e-3); loss_fn = nn.BCELoss()
for ep in range(30):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        idx = perm[i:i + 32]
        opt.zero_grad(); loss = loss_fn(net(X[idx]), y[idx]); loss.backward(); opt.step()
    print(f"epoca {ep + 1}  loss {loss.item():.3f}")
torch.save(net.state_dict(), "models/stain_cnn.pt")
