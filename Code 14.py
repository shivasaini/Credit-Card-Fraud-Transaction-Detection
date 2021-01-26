#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# # Preprocessing

# In[2]:


# COLUMNS WITH STRINGS
str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']


# In[3]:


# FIRST 53 COLUMNS
cols = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 
        'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 
        'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 
        'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 
        'M6', 'M7', 'M8', 'M9']


# In[4]:


# V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
# https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
v =  [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65, 67, 68, 
      70, 76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136, 138, 
      139, 142, 147, 156, 162, 165, 160, 166, 178, 176, 173, 182, 187, 203, 205, 207, 215, 169, 171, 175, 180, 185, 
      188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235, 240, 258, 257, 253, 252, 260, 261, 264, 266, 267, 274, 
      277, 220, 221, 234, 238, 250, 271, 294, 284, 285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289, 
      296, 301, 314]


# In[5]:


cols += ['V'+str(x) for x in v]
dtypes = {}
for x in range(1,10):
    cols += ['id_0'+str(x)]
for x in range(10,34):
    cols += ['id_'+str(x)]
# for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]: 
for c in cols:
    dtypes[c] = 'float32'
for c in str_type:
    dtypes[c] = 'object'


# In[6]:


# LOAD TRAIN
X_train = pd.read_csv('train_transaction.csv',index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])
train_id = pd.read_csv('train_identity.csv',index_col='TransactionID', dtype=dtypes)
X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)


# In[7]:


# LOAD TEST
X_test = pd.read_csv('test_transaction.csv',index_col='TransactionID', dtype=dtypes, usecols=cols)
test_id = pd.read_csv('test_identity.csv',index_col='TransactionID', dtype=dtypes)
X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)


# In[8]:


# TARGET
y_train = X_train['isFraud'].copy()
del train_id, test_id, X_train['isFraud']
x = gc.collect()


# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# In[10]:


null_cols = []
for col in X_train.columns:
    if (X_train[col].isnull().sum() / X_train.shape[0]) > 0.9:
        many_null_cols.append(col)


# In[11]:


top_cols = []
for col in X_train.columns:
    temp = X_train[col].value_counts(dropna=False, normalize=True)
    if (temp.values[0]) > 0.9:
        big_top_value_cols.append(col)


# In[12]:


single_cols = []
for col in X_train.columns:
    if X_train[col].nunique() <= 1:
        single_cols.append(col)
single_cols_test = []
for col in X_test.columns:
    if X_test[col].nunique() <= 1:
        single_cols_test.append(col)


# In[13]:


cols_to_drop = list(set(null_cols + top_cols + single_cols + single_cols_test))

df_train = X_train.drop(cols_to_drop, axis=1)
df_test = X_test.drop(cols_to_drop, axis=1)


# In[14]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 
          'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 
          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 
          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 
          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 
          'juno.com': 'other', 'icloud.com': 'apple'}


# In[15]:


us_emails = ['gmail', 'net', 'edu']


# In[16]:


# Reference: https://www.kaggle.com/tolgahancepel/lightgbm-single-model-and-feature-engineering
for c in ['P_emaildomain', 'R_emaildomain']:
    df_train[c + '_bin'] = df_train[c].map(emails)
    df_test[c + '_bin'] = df_test[c].map(emails)
    
    df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])
    df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])
    
    df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


# In[17]:


from sklearn.preprocessing import minmax_scale


# In[18]:


id_n_d_num_cols = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'D1', 'D3', 'D5', 
               'D6', 'D8', 'D9', 'D10', 'D11', 'D13', 'D14', 'D4', 'D15']
    
for col in id_n_d_num_cols:
    df_train[col] = (minmax_scale(df_train[col], feature_range=(0,1)))
    df_train[col] = df_train[col].fillna(-1)


# In[19]:


from sklearn import preprocessing


# In[20]:


for f in df_train.columns:
    if df_train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values)) 


# In[21]:


tr = df_train.merge(y_train, how='left', left_index=True, right_index=True)


# In[25]:


categ = df_train.select_dtypes(include=['object'])


# In[26]:


no_missing_cat = categ.fillna('Empty')


# In[27]:


numerical = df_train._get_numeric_data()
no_missing_numer = numerical.fillna(-1)


# In[28]:


train=pd.concat([no_missing_numer,no_missing_cat],axis=1) 


# In[29]:


from sklearn.linear_model import LogisticRegression


# # SMOTE

# In[31]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(train, y_train)


# In[32]:


Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(train, y_train, test_size=0.33, random_state=42)


# # Logistic Regression With Balancing

# In[34]:


clf = LogisticRegression().fit(Xs_train, Ys_train)


# In[35]:


y_predlg = clf.predict(Xs_test)
accuracylg = accuracy_score(Ys_test, y_predlg)
print("Accuracy of Logistic Regression on Testing Data (Balanced)" , (accuracylg * 100.0))


# In[36]:


y_predlgt = clf.predict(Xs_train)
accuracylgt = accuracy_score(Ys_train, y_predlgt)
print("Accuracy of Logistic Regression on Training Data (Balanced)", (accuracylgt * 100.0))


# In[70]:


plt.figure(1)
plt.title('Confusion Matrix on Testing Data of Logistic Regression (Balanced)')
sns.heatmap(confusion_matrix(Ys_test,y_predlg), fmt='d',cmap='YlGnBu', annot=True)
plt.show()


# In[38]:


y_score_rfd = clf.predict_proba(Xs_test)[:,-1]


# In[71]:


from sklearn.metrics import roc_curve,auc
fpr_rf, tpr_rf, _ = roc_curve(Ys_test, y_score_rfd)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(2, figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))


plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve of Logistic Regression', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# # Logistic Regression Without Balancing

# In[40]:


clf_w = LogisticRegression().fit(X_train, Y_train)


# In[41]:


y_predlg_w = clf_w.predict(X_test)
accuracylg_w = accuracy_score(Y_test, y_predlg_w)
print("Accuracy of Logistic Regression on Testing Data (Imbalanced)" , (accuracylg_w * 100.0))


# In[42]:


y_predlgt_w = clf_w.predict(X_train)
accuracylg_wt = accuracy_score(Y_train, y_predlgt_w)
print("Accuracy of Logistic Regression on Training Data (Imbalanced)", (accuracylg_wt * 100.0))


# In[43]:


plt.figure(3)
plt.title('Confusion Matrix on Testing Data of Logistic Rogression (Imbalanced)')
sns.heatmap(confusion_matrix(Y_test,y_predlg_w), fmt='d',cmap='YlGnBu',annot=True)
plt.show()


# # MLP With Balancing

# In[44]:


from sklearn.neural_network import MLPClassifier


# In[45]:


mlp = MLPClassifier(hidden_layer_sizes=(128,72), max_iter=100, batch_size=50000, activation = 'relu', 
                    learning_rate_init=0.01, learning_rate="constant", verbose=True)


# In[46]:


mlp.fit(Xs_train, Ys_train)


# In[47]:


y_pred_mlp = mlp.predict(Xs_test)
accuracymlp = accuracy_score(Ys_test, y_pred_mlp)
print("Accuracy of MLP on Testing Data (Balanced)" , (accuracymlp * 100.0))


# In[48]:


y_pred_mlpt = mlp.predict(Xs_train)
accuracymlpt = accuracy_score(Ys_train, y_pred_mlpt)
print("Accuracy of MLP on Training Data (Balanced)" , (accuracymlpt * 100.0))


# In[49]:


plt.figure(4)
plt.title('Confusion Matrix of Testing Data of MLP (Balanced)')
sns.heatmap(confusion_matrix(Ys_test, y_pred_mlp), fmt='d', cmap='YlGnBu', annot=True)
plt.show()


# In[50]:


y_score_mlp = mlp.predict_proba(Xs_test)[:,-1]


# In[68]:


fpr_rf, tpr_rf, _ = roc_curve(Ys_test, y_score_mlp)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(5, figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))


plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve of MLP', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# # MLP Without Balancing

# In[52]:


mlp_w = MLPClassifier(hidden_layer_sizes=(128,72), max_iter=100, batch_size=20000, activation = 'relu', 
                    learning_rate_init=0.01, learning_rate="constant", verbose=True)


# In[53]:


mlp_w.fit(X_train, Y_train)


# In[54]:


y_pred_mlp_w = mlp_w.predict(X_test)
accuracymlp_w = accuracy_score(Y_test, y_pred_mlp_w)
print("Accuracy of MLP on Testing Data (Imbalanced)" , (accuracymlp_w * 100.0))


# In[55]:


y_pred_mlpt_w = mlp_w.predict(X_train)
accuracymlpt_w = accuracy_score(Y_train, y_pred_mlpt_w)
print("Accuracy of MLP on Training Data (Imbalanced)" , (accuracymlpt_w * 100.0))


# In[56]:


plt.figure(6)
plt.title('Confusion Matrix of Testing Data of MLP (Imbalanced)')
sns.heatmap(confusion_matrix(Y_test, y_pred_mlp_w), fmt='d', cmap='YlGnBu', annot=True)
plt.show()


# # XGBOOST With Weight Balance

# In[ ]:


from xgboost import XGBClassifier


# In[65]:


Y_train.value_counts()


# In[66]:


model = XGBClassifier(scale_pos_weight=381944/13717)
model.fit(X_train,Y_train)


# In[67]:


y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred) ## Testing Accuracy
print('Accuracy of XGBoost on Testing Data (Balanced)', accuracy * 100.0)


# In[68]:


y_predt = model.predict(X_train) ## Training accuracy 

accuracyt = accuracy_score(Y_train, y_predt)
print('Accuracy of XGBoost on Training Data (Balanced)', accuracyt * 100.0)


# In[69]:


plt.figure(7)
plt.title('Confusion Matrix on Testing Data of XGBoost (Balanced)')
sns.heatmap(confusion_matrix(Y_test,y_pred), fmt='d',cmap='YlGnBu',annot=True)
plt.show()


# # XGBOOST Without Weight Balance

# In[70]:


model_w = XGBClassifier()
model_w.fit(X_train,Y_train)
y_pred_w = model_w.predict(X_test)


# In[71]:


accuracy_w= accuracy_score(Y_test, y_pred_w) ## Testing Accuracy
print('Accuracy of Testing Data on XGBoost (Imbalanced)', accuracy_w * 100.0)


# In[72]:


y_predt_w = model_w.predict(X_train) ## Training accuracy 

accuracyt_w = accuracy_score(Y_train, y_predt_w)
print('Accuracy of XGBoost on Training Data (Imbalanced)', accuracyt_w * 100.0)


# In[69]:


plt.figure(8)
plt.title('Confusion Matrix on Testing Data of XGBoost (Imbalanced)')
sns.heatmap(confusion_matrix(Y_test,y_pred_w), fmt='d',cmap='YlGnBu',annot=True)
plt.show()


# # ROC of XGBOOST (Balanced)

# In[74]:


y_score_rf1 = model.predict_proba(X_test)[:,-1]


# In[75]:


from sklearn.metrics import roc_curve,auc
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score_rf1)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(9, figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))


plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve of XGBoost', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# # LGBM With Weight Balance

# In[57]:


Lgb = lgb.LGBMClassifier(is_unbalance=True)


# In[58]:


Lgb.fit(X_train,Y_train)


# In[59]:


y_pred1 = Lgb.predict(X_test)
accuracy1 = accuracy_score(Y_test, y_pred1) ## Testing Accuracy
print("Accuracy of Testing Data on LGBM (Balanced)", (accuracy1 * 100.0))


# In[60]:


y_pred1t = Lgb.predict(X_train)
accuracy1t = accuracy_score(Y_train, y_pred1t) ## Testing Accuracy
print("Accuracy of Training Data on LGBM (Balanced)", (accuracy1t * 100.0))


# In[61]:


plt.figure(10)
plt.title('Confusion Matrix on Testing Data of LGBM (Balanced)')
sns.heatmap(confusion_matrix(Y_test,y_pred1), fmt='d',cmap='YlGnBu',annot=True)
plt.show()


# # LGBM Without Weight Balance

# In[62]:


Lgb_w = lgb.LGBMClassifier().fit(X_train,Y_train)


# In[63]:


y_pred1_w = Lgb_w.predict(X_test)
accuracy1_w = accuracy_score(Y_test, y_pred1_w) ## Testing Accuracy
print("Accuracy of Testing Data on LGBM (Imbalanced)", (accuracy1_w * 100.0))


# In[64]:


y_pred1_wt = Lgb_w.predict(X_train)
accuracy1_wt = accuracy_score(Y_train, y_pred1_wt) ## Training Accuracy
print("Accuracy of Training Data of LGBM (Imbalanced)", (accuracy1_wt * 100.0))


# In[65]:


plt.figure(11)
plt.title('Confusion Matrix on Testing Data of LGBM (Imbalanced)')
sns.heatmap(confusion_matrix(Y_test,y_pred1_w), fmt='d',cmap='YlGnBu',annot=True)
plt.show()


# # ROC Curve of LGBM

# In[66]:


y_score_rf2 = Lgb.predict_proba(X_test)[:,-1]


# In[67]:


fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score_rf2)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(12, figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))


plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve of LGBM', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# In[ ]:




