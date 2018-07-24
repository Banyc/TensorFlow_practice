import os
import const


train_xs = []
test_xs = []
all_xs = []




walk = os.walk(const.ffle)
for root, directories, file_names in walk:
    all_xs = file_names.copy()

fp_train = open(const.TRAIN_PATH)
while 1:
   line = fp_train.readline()
   if not line:
       break
   value = line.split(",")
   img = value[0]
   train_xs.append(img)
fp_train.close()

#求差集，在 all_xs 中但不在 train_xs 中
# https://blog.csdn.net/bitcarmanlee/article/details/51622263
test_xs = list(set(all_xs).difference(set(train_xs)))
print(test_xs)

fp_test = open(const.TEST_PATH, "w")
for test_x in test_xs:
    test_y = test_x.split('_')[-1].split('.')[0]
    fp_test.write("%s,%s\n" % (test_x, test_y))
fp_test.close()
