"""
Calculates ECE

Example call: python calc_ece.py --scores_csv model_scores/deberta_v3_large/open_ai_summarize_from_feedback/english_original_scores.csv  --reliability_diagram_path reliable.png --reliability_diagram_title "SummarizeFromFeedback DeBERTa"
"""
import numpy as np
import fire
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def calc_bins(y_true, preds, confs, num_bins=10):
  # Assign each prediction to a bin
  bins = np.linspace(1.0 / num_bins, 1, num_bins)
  binned = np.digitize(confs, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = np.mean(y_true[binned==bin] == preds[binned == bin])
      bin_confs[bin] = np.mean(confs[binned==bin])

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(y_true, preds, confs):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, preds, confs)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  overall_acc = np.mean(y_true == preds)
  return ECE, MCE, overall_acc
  
def draw_reliability_graph(y_true, preds, confs, save_path, title=None):
  ECE, MCE, overall_acc = get_metrics(y_true, preds, confs)
  bins, _, bin_accs, _, _ = calc_bins(y_true, preds, confs)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=0.7, edgecolor='red', color='purple')
  plt.bar(bins, np.minimum(bins,bin_accs), width=0.1, alpha=1, edgecolor='black', color='b')

  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  #MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  #acc_patch = mpatches.Patch(color='orange', label='Overall Accuracy = {:.2f}%'.format(overall_acc*100))
  plt.legend(handles=[ECE_patch])
  if title is not None:
    plt.title(title)
  fig.savefig(save_path)
  
def main(scores_csv, reliability_diagram_path=None, reliability_diagram_title=None):
  scores = pd.read_csv(scores_csv)
  assert scores.shape[1] == 2
  assert scores.columns.to_list() == ['chosen', 'rejected']
  scores = softmax(scores.values, axis=1)
  y_true = np.array([0] * len(scores)) # chosen is always class 0
  preds = scores.argmax(axis=1)
  confs = scores[np.arange(len(scores)), preds]
  ece, mce, acc = get_metrics(y_true=y_true, preds=preds, confs=confs)
  print(f"ECE = {round(ece*100, 2)}%, accuracy = {round(acc*100, 2)}%")

  if reliability_diagram_path is not None:
    draw_reliability_graph(y_true, preds, confs, reliability_diagram_path, title=reliability_diagram_title)
  

if __name__ == '__main__':
  fire.Fire(main)