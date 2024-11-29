import torch
from torch import nn
from tqdm import tqdm
from baseclasses.baseclass import BaseTrainer
from architectures.simple_mnistNet import SimpleMNISTNet

class SimpleMNISTTrainer(BaseTrainer):
    def __init__(self, args):
        super(SimpleMNISTTrainer, self).__init__(args)
        self.model = SimpleMNISTNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.epochs = 1000
        self.device = args.device
        self.model.to(self.device)

    def train_setup(self, train_data, valid_data=None):
        self.train_data = train_data
        self.valid_data = valid_data

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc='MNIST training'):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_data, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # optimizer run
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # log and print
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

    def test_setup(self, test_data):
        self.test_data = test_data

    def test(self):
        self.model.eval()  
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                # forward run
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # print
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy:.2f}%")

