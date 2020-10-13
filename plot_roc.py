import sklearn.metrics as metrics
# method I: plt
import matplotlib.pyplot as plt
import numpy as np



lbl = np.load('./data/pr_pr_pix2pix_lbl.npy')
dist = np.load('./data/pr_pr_pix2pix_dist.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label='Fprint vs Fprint, cGAN, AUC = %0.6f' % roc_auc)


# pr_ph train

lbl = np.load('./data/pr_pr_pix2pix_verifier3_lbl.npy')
dist = np.load('./data/pr_pr_pix2pix_verifier3_dist.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)


plt.plot(fpr, tpr, 'r', label='Fprint vs Fprint, cGAN+Verifier, AUC = %0.6f' % roc_auc)



plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
exit()

# pr_ph test

lbl = np.load('data/GanLoss_CLoss/pr_ph_lbl_test.npy')
dist = np.load('data/GanLoss_CLoss/pr_ph_dist_test.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label='Fprint vs Fphoto, test, AUC = %0.6f' % roc_auc)


# pr_ph train

lbl = np.load('./data/pr_ph_lbl_train.npy')
dist = np.load('./data/pr_ph_dist_train.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)


plt.plot(fpr, tpr, '--b', label='Fprint vs Fphoto, train, AUC = %0.6f' % roc_auc)


# pr_pr test

lbl = np.load('./data/pr_pr_lbl_test.npy')
dist = np.load('./data/pr_pr_dist_test.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'r', label='Fprint vs Fprint, test, AUC = %0.6f' % roc_auc)


# pr_pr train

lbl = np.load('./data/pr_pr_lbl_train.npy')
dist = np.load('./data/pr_pr_dist_train.npy')

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)


plt.plot(fpr, tpr, '--r', label='Fprint vs Fprint, train, AUC = %0.6f' % roc_auc)


plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim([0, 0.4])
plt.ylim([0.6, 1])
plt.show()