import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, epochs=5, learning_rate=1e-3):
    """"""
    torch.manual_seed(42)
    criterion= nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    outputs = []
    for ep in range(epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch: {} - Loss: {}".format(ep + 1, float(loss)))
        outputs.append((epochs, img, recon))
    return outputs

