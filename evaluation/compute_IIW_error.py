import os 
import whdr
import math
import string

inputPath = '/raid/qingnan/codes/intrinsic/IIW_combine/'
dataPath = "/raid/qingnan/data/iiw-dataset/data/"

score = 0
count = 0

for file in os.listdir(inputPath):
	if file.endswith('-R.png'):
		imagePath = inputPath + file
		labelPath = dataPath + file[:-6] + '.json'

		temp = whdr.whdr_final(imagePath,labelPath)
		score = score + temp
		count = count + 1

print('average')
print(score/count)
