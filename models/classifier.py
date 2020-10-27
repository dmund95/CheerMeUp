import torchvision.models as models
import torch.nn as nn

class Extractor(nn.Module):
	def __init__(self):
		super(Extractor, self).__init__()
		self.model = models.alexnet(pretrained=True)

	def forward(self,x):
		return self.model.features(x)

class Classifier(nn.Module):
	def __init__(self, num_classes):
		super(Classifier, self).__init__()

		self.fc1 = nn.Linear(256, 128)
		self.relu = nn.ReLU(inplace=True)
		self.batchnorm = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = self.fc2(self.batchnorm(self.relu(self.fc1(x))))
		return x