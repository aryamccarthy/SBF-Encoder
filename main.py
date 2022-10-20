import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import gc
import time
import torchvision.transforms as transforms

from SBF_arch import count_trainable_parameters, ViT, BViTThree


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct


def train(trainloader, testloader, model, loss_fn, optimizer, device, num):
    size = len(trainloader.dataset)
    model.train()
    modelTrainHistory = []
    modelTestHistory = []
    timeNow = time.time_ns()
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            print(time.time_ns() - timeNow)

            loss, current = loss.item(), batch * len(X)

            modelTrainHistory.append(loss)

            testLoss = test(testloader, model, loss_fn, device)

            modelTestHistory.append(testLoss)
            model.train()

            print(f"Model {num} \n test loss: {testLoss:>7f} loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            timeNow = time.time_ns()
    return modelTrainHistory, modelTestHistory


def run(trainset, testset, model, optimizer, device, epochs, loss_fn, iteration, num):
    trainH, testH = [], []
    for t in range(epochs):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=1)
        gc.collect()
        print(f"Epoch {t + 1}\n-------------------------------")
        trainT, testT = train(trainloader, testloader, model, loss_fn, optimizer, device, num)
        trainH = trainH + trainT
        testH = testH + testT

    with open(f'modelVIT{num}TestHist{iteration}.txt', 'w') as fp:
        for item in testH:
            # write each item on a new line
            fp.write("%s\n" % item)


def main(num):
    epochs = 20
    device = "cuda"
    print(device)
    learningrate = 10e-5

    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    for i in range(5):
        print(f"-------starting {i + 1}th iteration!-------")
        ms512 = ViT(in_channels=3,
                    patch_size=8,
                    emb_size=512,
                    img_size=32,
                    depth=6,
                    n_classes=10).to(device)
        ls512 = nn.CrossEntropyLoss()
        optimizerMs512 = torch.optim.AdamW(ms512.parameters(), lr=learningrate)

        m0512 = ViT(in_channels=3,
                    patch_size=8,
                    emb_size=512,
                    img_size=32,
                    depth=12,
                    n_classes=10).to(device)
        l0512 = nn.CrossEntropyLoss()
        optimizerM0512 = torch.optim.AdamW(m0512.parameters(), lr=learningrate)

        #         m1 = BViTOne(in_channels = 3,
        #                 patch_size = 8,
        #                 emb_size = 128,
        #                 img_size = 32,
        #                 depth = 12,
        #                 n_classes = 10).to(device)
        #         l1 = nn.CrossEntropyLoss()
        #         optimizerM1 = torch.optim.AdamW(m1.parameters(), lr=learningrate)

        #         m2 = BViTTwo(in_channels = 3,
        #                 patch_size = 8,
        #                 emb_size = 128,
        #                 img_size = 32,
        #                 depth = 12,
        #                 n_classes = 10).to(device)
        #         l2 = nn.CrossEntropyLoss()
        #         optimizerM2 = torch.optim.AdamW(m2.parameters(), lr=learningrate)

        m3512 = BViTThree(in_channels=3,
                          patch_size=8,
                          emb_size=512,
                          img_size=32,
                          depth=12,
                          n_classes=10).to(device)
        l3512 = nn.CrossEntropyLoss()
        optimizerM3512 = torch.optim.AdamW(m3512.parameters(), lr=learningrate)

        print(num)
        print(f"Number of parameters: {count_trainable_parameters(locals()[f'm{num}'])}")
        run(trainset, testset, locals()[f"m{num}"], locals()[f"optimizerM{num}"], device, epochs, locals()[f"l{num}"],
            i, num)


if __name__ == '__main__':
    for i in ['s512']:
        main(i)
