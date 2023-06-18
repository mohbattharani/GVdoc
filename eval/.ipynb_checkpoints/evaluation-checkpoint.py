import numpy as np
from numpy import exp
import math, json, sys, os
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

from statistics import mean
from pprint import pprint
from sklearn.metrics import roc_auc_score

# If DCS, skip ood_metric as it is not installed on DCS
import platform
processor = platform.processor() 
if (not processor in ["ppc64le"]):
    from ood_metrics import fpr_at_95_tpr


# https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def cm_analysis(y_true, y_pred, labels, ymap=None):
    
    """
    
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    s = 10 if len(labels) <15 else 20
    figsize=(s, s)
    
    if ymap is not None:
        # change category codes or labels to new labels 
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    # calculate a confusion matrix with the new labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # calculate row sums (for calculating % & plot annotations)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    # calculate proportions
    cm_perc = cm / cm_sum.astype(float) * 100
    # empty array for holding annotations for each cell in the heatmap
    annot = np.empty_like(cm).astype(str)
    # get the dimensions
    nrows, ncols = cm.shape
    # cycle over cells and create annotations for each cell
    for i in range(nrows):
        for j in range(ncols):
            # get the count for the cell
            c = cm[i, j]
            # get the percentage for the cell
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                # convert the proportion, count, and row sum to a string with pretty formatting
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    # convert the array to a dataframe. To plot by proportion instead of number, use cm_perc in the DataFrame instead of cm
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    # create empty figure with a specified size
    fig, ax = plt.subplots(figsize=figsize)
    # plot the data using the Pandas dataframe. To change the color map, add cmap=..., e.g. cmap = 'rocket_r'
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    ##plt.savefig(filename)
    #plt.show()
    return fig
    
def compute_plot_cm (file):
    with open(file, 'r') as f:
        logs = json.load(f) 
    
    pred_labels = []
    true_labels = []
    for item in logs:
        predicted = item['predicted_label']
        actual = item['true_label']
        pred_labels.append(predicted)
        true_labels.append(actual)
    
    labels = list (set(true_labels + pred_labels))  #+ list (set(pred_labels)) 

    fig = cm_analysis(true_labels, pred_labels, labels)
    plt.title(file.replace ("json", "jpg"))
    
    return fig 

def energy2(vector, T = 1):
    e = 1 * T * math.log (exp( np.array(vector)/T).sum())
    return e 

def energy(logits, t):
    _sum = 0
    for logit in logits:
        _sum += math.exp(-1*logit / t)
    return t * math.log(_sum)


class Accuracy ():
    def __init__(self) -> None:
        pass

    def computer_macro_acc (self, logs):
        scores = {}
        
        for item in logs:
            predicted = item['predicted_label']
            actual = item['true_label']
            
            if actual not in scores:
                scores[actual] = []
            if predicted == actual:
                scores[actual].append(1)
            else:
                scores[actual].append(0)
        
        for label, hitlist in scores.items():
            acc = sum(hitlist) / len(hitlist)
            scores[label] = acc
        macro_score = mean(scores.values())
        return {"macro_acc_scores": scores, "macro_acc": macro_score}

    def compute_acc(self, logs):
        hits = 0
        labels = []
        true_labels = []
        for item in logs:
            predicted = item['predicted_label']
            actual = item['true_label']
            labels.append(predicted)
            true_labels.append(actual)
            if predicted == actual:
                hits += 1
        
        acc = hits / len(logs)

        return {"micro_acc": acc}

    def __call__(self, logs):
        macro = self.computer_macro_acc (logs)
        micro = self.compute_acc (logs)

        return {"macro": macro, "micro": micro}

class EvaluationOOD ():
    def __init__(self) -> None:
        pass 
    
    def compute_auc(self, pos_confidences, neg_confidences):
            pos = [1]*len(pos_confidences)
            neg = [0]*len(neg_confidences)
            y = pos + neg
            confidences = pos_confidences + neg_confidences
            score = roc_auc_score(y, confidences)
            return score

    # computes macro AUC on in-domain versus out-of-domain output for Energy
    def compute_macro_auc_energy(self, pos, neg, tt):
  
        labels = list(set([item['predicted_label'] for item in neg]))
        scores = {}
        for label in labels:
            pos_confidences = []
            for item in pos:
                true_label = item['true_label']
                if true_label == 'article':
                    true_label = 'news_article'
                if true_label == 'publication':
                    true_label == 'scientific_publication'
                if true_label == label:
                    pos_confidences.append(energy(item['logits'], tt))
            neg_confidences = []
            for item in neg:
                if item['predicted_label'] == label:
                    neg_confidences.append(energy(item['logits'], tt))
            #print('working on: {}'.format(label))
            if len(pos_confidences) < 1 or len(neg_confidences) < 1:
                #print('skipping')
                continue
            scores[label] =self.compute_auc(pos_confidences, neg_confidences)
        

        return mean(scores.values())
    
    # computes macro auc for in-domain versus out-of-domain data for softmax/MSP
    def compute_macro_auc_msp(self, pos, neg):
        labels = list(set([item['predicted_label'] for item in neg]))
        #print(labels)
        scores = {}
        for label in labels:
            pos_confidences = []
            for item in pos:
                true_label = item['true_label']
                if true_label == 'article':
                    true_label = 'news_article'
                if true_label == 'publication':
                    true_label == 'scientific_publication'
                if true_label == label:
                    pos_confidences.append(float(item['confidence']))
            neg_confidences = []
            for item in neg:
                if item['predicted_label'] == label:
                    neg_confidences.append(float(item['confidence']))
            #print('working on: {}'.format(label))
            if len(pos_confidences) < 1 or len(neg_confidences) < 1:
                #print('skipping')
                continue
            scores[label] = self.compute_auc(pos_confidences, neg_confidences)
        #print(scores)
        #print(mean(scores.values()))
        return  mean(scores.values())

    # computes micro AUC on in-domain versus out-of-domain data for softmax/MSP
    def compute_micro_auc_msp (self, _id, _od):
        _id = [float(x['confidence']) for x in _id]
        _od = [float(x['confidence']) for x in _od]
        micro_auc_msp = self.compute_auc (_id, _od)

        return micro_auc_msp
    
    # computes micro AUC on in-domain versus out-of-domain data for Energy-based confidence method
    def compute_micro_auc_energy (self, _id, _od):
        micro_auc_energy = {}

        for t in [1]:
            _id_ = [energy(x['logits'], t) for x in _id]
            _od_ = [energy(x['logits'], t) for x in _od]
            micro_auc = self.compute_auc (_id_, _od_)
            micro_auc_energy [t] = micro_auc

        return micro_auc


    def compute_fpr95(self, pos_confidences, neg_confidences):
        pos = [1]*len(pos_confidences)
        neg = [0]*len(neg_confidences)
        y = pos + neg
        confidences = pos_confidences + neg_confidences
        score = fpr_at_95_tpr(confidences, y)
        return score

    def compute_macro_fpr95_msp(self, pos, neg):
        labels = list(set([item['predicted_label'] for item in neg]))
        scores = {}
        for label in labels:
            pos_confidences = []
            for item in pos:
                true_label = item['true_label']
                if true_label == 'article':
                    true_label = 'news_article'
                if true_label == 'publication':
                    true_label == 'scientific_publication'
                if true_label == label:
                    pos_confidences.append(float(item['confidence']))
            neg_confidences = []
            for item in neg:
                if item['predicted_label'] == label:
                    neg_confidences.append(float(item['confidence']))
            if len(pos_confidences) < 1 or len(neg_confidences) < 1:
                continue
            scores[label] = self.compute_fpr95(pos_confidences, neg_confidences)
        
        return mean(scores.values())

    def compute_micro_fpr95_msp(self, _id, _od):
        id = [float(x['confidence']) for x in _id]
        od = [float(x['confidence']) for x in _od]
        return self.compute_fpr95(id, od)
    
    def compute_micro_fpr95_energy(self, _id, _od):
        t = 1
        _id_ = [energy(x['logits'], t) for x in _id]
        _od_ = [energy(x['logits'], t) for x in _od]

        return self.compute_fpr95(_id_, _od_)

    def compute_macro_fpr95_energy(self, pos, neg):
        tt = 1
        labels = list(set([item['predicted_label'] for item in neg]))
        scores = {}
        for label in labels:
            pos_confidences = []
            for item in pos:
                true_label = item['true_label']
                if true_label == 'article':
                    true_label = 'news_article'
                if true_label == 'publication':
                    true_label == 'scientific_publication'
                if true_label == label:
                    pos_confidences.append(energy(item['logits'], tt))
            neg_confidences = []
            for item in neg:
                if item['predicted_label'] == label:
                    neg_confidences.append(energy(item['logits'], tt))
            #print('working on: {}'.format(label))
            if len(pos_confidences) < 1 or len(neg_confidences) < 1:
                #print('skipping')
                continue
            scores[label] =self.compute_fpr95(pos_confidences, neg_confidences)
        

        return mean(scores.values()) 
    
   

    def __call__(self, _id, _od):
        
        micro_auc_msp = self.compute_micro_auc_msp (_id, _od)
        macro_auc_msp = self.compute_macro_auc_msp(_id, _od)
        micro_auc_energy = self.compute_micro_auc_energy(_id, _od)
        macro_auc_energy = self.compute_macro_auc_energy(_id, _od, 1)
        macro_fpr95_msp = self.compute_macro_fpr95_msp  (_id, _od)
        micro_fpr95_msp = self.compute_micro_fpr95_msp  (_id, _od)
        macro_fpr95_energy = self.compute_macro_fpr95_energy (_id, _od)
        micro_fpr95_energy = self.compute_micro_fpr95_energy  (_id, _od)

        #text = "micro_auc_msp: %0.3f macro_auc_msp: %0.3f micro_auc_energy:%0.3f macro_auc_energy: %0.3f"%(micro_auc_msp, macro_auc_msp, micro_auc_energy, macro_auc_energy)
        #text2 = " micro_fpr95_msp: %0.3f macro_fpr95_msp: %0.3f micro_fpr95_energy:%0.3f macro_fpr95_energy: %0.3f"%(micro_fpr95_msp, macro_fpr95_msp, micro_fpr95_energy, macro_fpr95_energy)

        #return text + text2 

        return {
                "micro_auc_msp": micro_auc_msp,
                "macro_auc_msp": macro_auc_msp,
                "micro_auc_energy": micro_auc_energy,
                "macro_auc_energy": macro_auc_energy, 
                "micro_fpr95_msp": micro_fpr95_msp,  
                "macro_fpr95_msp": macro_fpr95_msp,  
                "micro_fpr95_energy": micro_fpr95_energy,
                "macro_fpr95_energy": macro_fpr95_energy,
                
             }
        


class SCORES ():
    def __init__(self) -> None:
        self.compute_accuracy_scores = Accuracy()
        self.eval_ood = EvaluationOOD()

    def print_ood (self, scores_ood):
        s = scores_ood
        print ("|   AUC: MSP    |   AUC:Energy  ||   FPR95: MSP   |  FPR95:Energy |")
        print ("| Micro | Macro | Micro | Macro ||  Micro | Macro | Micro | Macro |")
        print ("| %2.3f | %2.3f | %2.3f | %2.3f ||  %2.3f | %2.3f | %2.3f | %2.3f | "%(s['micro_auc_msp'], s['macro_auc_msp'], s['micro_auc_energy'], s['macro_auc_energy'], s['micro_fpr95_msp'], s['macro_fpr95_msp'], s['micro_fpr95_energy'], s['macro_fpr95_energy']))
        
        
    def __call__(self, file_id, file_od = None, do_each = True):
        with open(file_id, 'r') as f:
            _id = json.load(f) 
        
        print (file_id.split ("/")[-1])
        acc_scores1 = self.compute_accuracy_scores (_id)
        
        if (do_each):
            pprint (acc_scores1)
        else:
            text = "micro_acc:%0.4f  macro_acc:%0.4f "%(acc_scores1['micro']['micro_acc'], acc_scores1['macro']['macro_acc'])
            print (text)
        
        if (not file_od is None):
            ood_file_name = file_od.split ("/")[-1]
            print (file_od.split ("/")[-1])
            with open(file_od, 'r') as f:
                _od = json.load(f)
        
            scores_ood = self.eval_ood (_id, _od)
            
            self.print_ood (scores_ood)




# ============================================================================
# ============================================================================

# ============================================================================
# ============================================================================

#config = "../config/rvlcdip_ood/rvlcdip_o.json"
#test = Test (config, n, labels=rvlcdip_labels)
#logs = test.test ()
#logs = "logs/dropoutrvlcdip_n_GraphLayoutv6_10.json"
#pprint (scores (logs))


# ============================================================================
# ============================================================================


