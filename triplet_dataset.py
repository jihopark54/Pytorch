import random

filename = 'list_landmarks_inshop.txt'

id_list = []
with open(filename, 'r') as f:
	lines = f.readlines()
	i = 0
	
	for line in lines:
		if line == "":
			continue
		if i < 2:
			i += 1
			continue
		names = line.split()
		attr = names[0].split('/')
		id_list.append(attr[1:5] + [str(i)])
		i += 1
	
class_set = set()
for item in id_list:
	class_set.add(item[0]  + '_' + item[1]) # classify gender
	#class_set.add(item[1]) # don't classify gender

class_list = list(class_set)
class_num = len(class_list)
Classes = [[] for _ in range(class_num)]

for item in id_list:
	idx = class_list.index(item[0] + '_' + item[1])
	Classes[idx].append(item)

item_set = set()
for item in id_list:
	item_set.add(item[2] + '_' + item[3].split('_')[0])

item_list = list(item_set)
item_list_num = len(item_list)
Items = [[] for _ in range(item_list_num)]

for item in id_list:
	idx = item_list.index(item[2] + '_' + item[3].split('_')[0])
	Items[idx].append(item)

print(item_list_num)

index = 0
for i in range(item_list_num):
	if len(Items[i]) == 1:
		index = i


train_list = []
train_num = 100000
train_count = 0


while train_num > train_count:
	anchor_class = random.randrange(item_list_num)
	while len(Items[anchor_class]) < 2:
		anchor_class = random.randrange(item_list_num)
	positive_class = anchor_class
	negative_class = random.randrange(item_list_num)
	while anchor_class == negative_class:
		negative_class = random.randrange(item_list_num)
	anchor_cloth = class_list.index(Items[anchor_class][0][0] + '_' + Items[anchor_class][0][1])
	negative_cloth = class_list.index(Items[negative_class][0][0] + '_' + Items[negative_class][0][1])
	while anchor_cloth != negative_cloth:
		negative_class = random.randrange(item_list_num)
		while anchor_class == negative_class:
			negative_class = random.randrange(item_list_num)
		negative_cloth = class_list.index(Items[negative_class][0][0] + '_' + Items[negative_class][0][1])

	anchor = random.randrange(len(Items[anchor_class]))
	positive = random.randrange(len(Items[positive_class]))
	while anchor == positive:
		positive = random.randrange(len(Items[positive_class]))
	negative = random.randrange(len(Items[negative_class]))

	try:
		train_list.index([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]]) #id
		
	except:
		train_list.append([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]])
		train_count += 1

print('step1')
train_count = 0

while train_num > train_count:
	anchor_class = random.randrange(item_list_num)
	while len(Items[anchor_class]) < 2:
		anchor_class = random.randrange(item_list_num)
	positive_class = anchor_class
	negative_class = random.randrange(item_list_num)
	while anchor_class == negative_class:
		negative_class = random.randrange(item_list_num)

	anchor = random.randrange(len(Items[anchor_class]))
	positive = random.randrange(len(Items[positive_class]))
	while anchor == positive:
		positive = random.randrange(len(Items[positive_class]))
	negative = random.randrange(len(Items[negative_class]))

	try:
		train_list.index([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]]) #id
		
	except:
		train_list.append([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]])
		train_count += 1

print('step2')

test_list =[]
test_num = 20000
test_count = 0

while test_num > test_count:
	anchor_class = random.randrange(item_list_num)
	while len(Items[anchor_class]) < 2:
		anchor_class = random.randrange(item_list_num)
	positive_class = anchor_class
	negative_class = random.randrange(item_list_num)
	while anchor_class == negative_class:
		negative_class = random.randrange(item_list_num)
	anchor_cloth = class_list.index(Items[anchor_class][0][0] + '_' + Items[anchor_class][0][1])
	negative_cloth = class_list.index(Items[negative_class][0][0] + '_' + Items[negative_class][0][1])
	while anchor_cloth != negative_cloth:
		negative_class = random.randrange(item_list_num)
		while anchor_class == negative_class:
			negative_class = random.randrange(item_list_num)
		negative_cloth = class_list.index(Items[negative_class][0][0] + '_' + Items[negative_class][0][1])

	anchor = random.randrange(len(Items[anchor_class]))
	positive = random.randrange(len(Items[positive_class]))
	while anchor == positive:
		positive = random.randrange(len(Items[positive_class]))
	negative = random.randrange(len(Items[negative_class]))

	try:
		test_list.index([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]]) #id
		
	except:
		test_list.append([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]])
		test_count += 1

print('step3')
test_count = 0

while test_num > test_count:
	anchor_class = random.randrange(item_list_num)
	while len(Items[anchor_class]) < 2:
		anchor_class = random.randrange(item_list_num)
	positive_class = anchor_class
	negative_class = random.randrange(item_list_num)
	while anchor_class == negative_class:
		negative_class = random.randrange(item_list_num)

	anchor = random.randrange(len(Items[anchor_class]))
	positive = random.randrange(len(Items[positive_class]))
	while anchor == positive:
		positive = random.randrange(len(Items[positive_class]))
	negative = random.randrange(len(Items[negative_class]))

	try:
		test_list.index([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]]) #id
		
	except:
		test_list.append([Items[anchor_class][anchor][4], Items[positive_class][positive][4], Items[negative_class][negative][4]])
		test_count += 1


print('step4')

with open('triplet_train.txt', 'w') as f:
	for item in train_list:
		f.write(item[0] + ',' + item[1] + ',' + item[2] + '\n')


with open('triplet_test.txt', 'w') as f:
	for item in test_list:
		f.write(item[0] + ',' + item[1] + ',' + item[2] + '\n')
