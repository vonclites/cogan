import os
import numpy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# delta1_list = [0.001]
delta1_list = [1]
# delta2_list = [1]
delta2_list = [2, 5, 10]
delta2_list = [2]
margin_list = [75]
feature_dims = [256]
# 75 > 25 > 50 > 100

# Delta_1:0.001, delta_2:1.000, margin:75, feature_dims:256
# The images don't really look good with this, they look better with higher delta values.

# Delta_1:6, delta_2:1.000, margin:75, feature_dims:256
# works pretty good

for delta1 in delta1_list:
    for delta2 in delta2_list:
        for margin in margin_list:
            for feature_dim in feature_dims:
                print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
                os.system("python train_cogan.py --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))

for delta1 in delta1_list:
    for delta2 in delta2_list:
        for margin in margin_list:
            for feature_dim in feature_dims:
                print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
                os.system("python test_cogan_nir_vis.py --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))

# for delta1 in delta1_list:
#     for delta2 in delta2_list:
#         for margin in margin_list:
#             for feature_dim in feature_dims:
#                 print("delta_1 %f delta_2 %f margin %d" % (delta1, delta2, margin))
#                 os.system("python SaveImages.py --batch_size 10 --delta_1 %f --delta_2 %f --margin %d --feat_dim %d" % (delta1, delta2, margin, feature_dim))


def loadRocCurve(file):
    tpr = list()
    fpr = list()
    skip = True
    with open(file, 'r') as lines:
        for line in lines:
            if not skip:
                tpr.append(line.split(",")[0].strip())
                fpr.append(line.split(",")[1].strip())
            skip = False
    return numpy.asanyarray(tpr, dtype=numpy.float32), numpy.asanyarray(fpr, dtype=numpy.float32)

fpr = dict()
tpr = dict()
roc_auc = dict()
result_dirs = dict()
i = 0
for delta1 in delta1_list:
    for delta2 in delta2_list:
        for margin in margin_list:
            for feature_dim in feature_dims:
                print("delta_1 %f delta_2 %f margin %d feat_dim %d" % (delta1, delta2, margin, feature_dim))
                dist = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_dist_test.npy")
                lbl = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_lbl_test.npy")
                fpr[i], tpr[i], threshold = metrics.roc_curve(lbl, dist)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                result_dirs[i] = "delta_1 %0.5f delta_2 %0.3f margin %d: %0.3f" % (delta1, delta2, margin, roc_auc[i])
                i = i + 1

delta1 = 0.001
delta2 = 1.0
margin = 75
feature_dim = 256
print("delta_1 %f delta_2 %f margin %d feat_dim %d" % (delta1, delta2, margin, feature_dim))
dist = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_dist_test.npy")
lbl = numpy.load("data/" + str(margin) + "_" + str(float(delta1)) + "_" + str(float(delta2)) + "_" + str(feature_dim) + "/pr_ph_lbl_test.npy")
fpr[i], tpr[i], threshold = metrics.roc_curve(lbl, dist)
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
result_dirs[i] = "delta_1 %0.5f delta_2 %0.3f margin %d: %0.3f" % (delta1, delta2, margin, roc_auc[i])
i = i + 1


fpr[i], tpr[i] = loadRocCurve("./data/SynthFprintvsFprint_DeepMatcher.csv")
result_dirs[i] = ("Synthetic Fprint vs Fprint (Siamese Network)")
roc_auc[i] = 0
i = i + 1

plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fingerprint vs Fingerphoto Matching ROC')
plt.plot([0, 1], [0, 1], 'k--')
for i in range(len(result_dirs)):
    plt.plot(fpr[i], tpr[i], label=result_dirs[i])
# plt.plot(fpr[i+1], tpr[i+1], label="No Change")
plt.legend(loc="lower right")
plt.show()
