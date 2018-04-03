import os

labels = list()
images = list()

with open('ILSVRC2012_validation_ground_truth.txt', 'r') as f:
	for line in f:
		labels.append(str(int(line) % 1000))

curdir = os.getcwd()
filenames = os.listdir(".")
for filename in filenames:
	if (filename.endswith('.JPEG')):
		images.append(os.path.join(curdir, filename))

with open('../ImageNet_train/val_map.txt', 'w') as f:
	for i in range(len(images)):
		f.write(images[i] + '	' + labels[i] + '\n')
print(images)
print(len(labels))
print(len(images))
