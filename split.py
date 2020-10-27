
import os
from random import shuffle
import csv

def getGenderDict():
	return {

		'aia' : ['female',0],
		'bonnie' : ['female',0],
		'jules' : ['male',1],
		'malcolm' : ['male',1],
		'mery' : ['female',0],
		'ray' : ['male',1]

	}

def getEmotionDict():
	return {

		'anger' : 0,
		'disgust' : 1,
		'fear' : 2,
		'joy' : 3,
		'neutral' : 4,
		'sadness' : 5,
		'surprise' : 6
	}

def getRows(image_names, dir, genderDict, emotionsDict):
	rows = []
	for n in image_names:
		char,emotion,_ = n.split('_')
		gender = genderDict[char][1]
		emotion = emotionsDict[emotion]
		rows.append([dir + '/' + n, gender, emotion])
	return rows


def createSplit():
	genderDict = getGenderDict()
	emotionsDict = getEmotionDict()
	parentDir = './data/FERG_DB_256'
	characters = os.listdir(parentDir)
	all_train_rows = []
	all_val_rows = []
	all_test_rows = []
	for c1 in characters:
		if(c1 not in genderDict):
			continue
		character_emotions = os.listdir(parentDir + '/' + c1)
		for c2 in character_emotions:
			if(c1 not in c2):
				continue
			all_images = os.listdir(parentDir + '/' + c1 + '/' + c2)
			shuffle(all_images)
			train = all_images[:int(0.8*len(all_images))]
			valAndTest = all_images[int(0.8*len(all_images)):]
			val = valAndTest[:int(len(valAndTest)/2)]
			test = valAndTest[int(len(valAndTest)/2):]
			cur_train = getRows(train, parentDir + '/' + c1 + '/' + c2, genderDict, emotionsDict)
			cur_val = getRows(val, parentDir + '/' + c1 + '/' + c2, genderDict, emotionsDict)
			cur_test = getRows(test, parentDir + '/' + c1 + '/' + c2, genderDict, emotionsDict)
			all_train_rows += cur_train
			all_val_rows += cur_val
			all_test_rows += cur_test

	trainWriter = csv.writer(open('./data/train.csv','w'))
	trainWriter.writerows(all_train_rows)

	valWriter = csv.writer(open('./data/val.csv','w'))
	valWriter.writerows(all_val_rows)

	testWriter = csv.writer(open('./data/test.csv','w'))
	testWriter.writerows(all_test_rows)

createSplit()



