import csv
import numpy as np
import sklearn.metrics as metrics
# method I: plt
import matplotlib.pyplot as plt
# field names
fields = ['fpr', 'tpr']

scores = np.load(
    '/media/aldb2/M2/python-projects/fingerprint/fingerphoto2printVerifier/data/pr_pr_pix2pix_verifier3_dist.npy')
labels = np.load(
    '/media/aldb2/M2/python-projects/fingerprint/fingerphoto2printVerifier/data/pr_pr_pix2pix_verifier3_lbl.npy')

# data rows of csv file
# rows = [['Nikhil', 'COE', '2', '9.0'],
#         ['Sanchit', 'COE', '2', '9.1'],
#         ['Aditya', 'IT', '2', '9.3'],
#         ['Sagar', 'SE', '1', '9.5'],
#         ['Prateek', 'MCE', '3', '7.8'],
#         ['Sahil', 'EP', '2', '9.1']]



fpr, tpr, threshold = metrics.roc_curve(labels, scores)
roc_auc = metrics.auc(fpr, tpr)


plt.plot(fpr, tpr, 'r', label='Fprint vs Fprint, cGAN+Verifier, AUC = %0.6f' % roc_auc)

plt.show()

print(scores.shape[0])
print(labels.shape[0])
rows = []
for i in range(fpr.shape[0]):
    # l = [scores[i], 1 - labels[i]]
    l = [fpr[i], tpr[i]]
    rows.append(l)

# name of csv file
filename = "./roc.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)
