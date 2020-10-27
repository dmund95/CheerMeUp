import torch
import os
import csv
import torch.nn as nn
import torch.optim as optim
from dataloader import get_loader, get_loader2
import torchvision.transforms as transforms
from models import classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extractor = classifier.Extractor().to(device)
emotion_classifier = classifier.Classifier(7).to(device)
gender_classifier = classifier.Classifier(2).to(device)

class_criterion = nn.CrossEntropyLoss()

params = list(list(extractor.parameters()) + list(emotion_classifier.parameters()) + list(gender_classifier.parameters()))
optimizer = torch.optim.Adam(params, lr=1e-3)

total_epochs = 100

baseDataPath = './data'
#data_path = f"{baseDataPath}/file_names.csv"
transform = transforms.Compose([transforms.Resize((64,64)), 
                                transforms.ToTensor()])

train_data = get_loader2(f"{baseDataPath}/train.csv", transform, 128, 4,shuffle=True)
val_data = get_loader2(f"{baseDataPath}/val.csv", transform, 128, 4,shuffle=True)
test_data = get_loader2(f"{baseDataPath}/test.csv", transform, 128, 4,shuffle=True)


min_val_loss = 10000.0
total_examples = 0
gender_correct = 0
emotion_correct = 0
for epoch in range(total_epochs):
	print('starting training')
	extractor.train()
	gender_classifier.train()
	emotion_classifier.train()

	for batch_idx, (image_data, gender_label, emotion_label) in enumerate(train_data):

		#optimizer.zero_grad()
		extractor.zero_grad()
		gender_classifier.zero_grad()
		emotion_classifier.zero_grad()
		total_examples += len(image_data)

		image_data,gender_label,emotion_label = image_data[:,:3, :, :].to(device), gender_label.to(device), emotion_label.to(device)

		feature = extractor.forward(image_data).squeeze()

		gender_output = gender_classifier.forward(feature)
		emotion_output = emotion_classifier.forward(feature)

		gender_loss = class_criterion(gender_output, gender_label)
		emotion_loss = class_criterion(emotion_output, emotion_label)
		total_loss = gender_loss + emotion_loss

		total_loss.backward()
		optimizer.step()

		max_index = gender_output.max(dim = 1)
		max_index = max_index[1]
		gender_correct += (max_index == gender_label.to(device)).sum().item()

		max_index = emotion_output.max(dim = 1)
		max_index = max_index[1]
		emotion_correct += (max_index == emotion_label.to(device)).sum().item()

		if(batch_idx % 20 == 0):
			print((epoch, batch_idx))
			print((gender_loss.item(), emotion_loss.item(), total_loss.item(), gender_correct/total_examples, emotion_correct/total_examples))

	print('starting validation')

	extractor.eval()
	gender_classifier.eval()
	emotion_classifier.eval()

	total_gender_loss = 0.0
	total_emotion_loss = 0.0
	total_examples = 0
	gender_correct = 0
	emotion_correct = 0

	for batch_idx, (image_data, gender_label, emotion_label) in enumerate(val_data):

		#optimizer.zero_grad()
		extractor.zero_grad()
		gender_classifier.zero_grad()
		emotion_classifier.zero_grad()
		total_examples += len(image_data)

		image_data,gender_label,emotion_label = image_data[:,:3, :, :].to(device), gender_label.to(device), emotion_label.to(device)

		feature = extractor.forward(image_data).squeeze()

		gender_output = gender_classifier.forward(feature)
		emotion_output = emotion_classifier.forward(feature)

		gender_loss = class_criterion(gender_output, gender_label)
		emotion_loss = class_criterion(emotion_output, emotion_label)

		max_index = gender_output.max(dim = 1)
		max_index = max_index[1]
		gender_correct += (max_index == gender_label.to(device)).sum().item()

		max_index = emotion_output.max(dim = 1)
		max_index = max_index[1]
		emotion_correct += (max_index == emotion_label.to(device)).sum().item()

		total_gender_loss += gender_loss.item()
		total_emotion_loss += emotion_loss.item()

	total_loss = total_emotion_loss + total_gender_loss

	if(total_loss < min_val_loss):
		min_val_loss = total_loss
		print((epoch, min_val_loss, gender_correct/total_examples, emotion_correct/total_examples))
		torch.save({
			'extractor' : extractor.state_dict(), 
			'gender_classifier' : gender_classifier.state_dict(), 
			'emotion_classifier' : emotion_classifier.state_dict(), 
			'optimizer' : optimizer.state_dict(),
			'stat' : (epoch, min_val_loss, gender_correct/total_examples, emotion_correct/total_examples)
			}, './results/classifier')
		print('saved')
		print('')














	



