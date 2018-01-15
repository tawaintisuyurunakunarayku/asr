import os, sys
import glob as gb
from my_flags import FLAGS
import numpy as np

np.random.seed(42)

if __name__ == "__main__":

  if len(sys.argv)!=3:
    print("args: <lang> <train_dir>")
    sys.exit(0)
    
  lang = sys.argv[1]
  train_dir = sys.argv[2]
  data_dir = os.path.expanduser("~/datasets/"+lang)
  output = open(os.path.join(train_dir,"train.subsample_indexes"),'w')

  unames = [os.path.basename(fn) for fn in gb.glob(os.path.join(data_dir,"train","*.txt"))]

  prop = 0.1 if lang=="eus" else 0.2 # 0.1 for eus, 0.2 for quz
  n_samples = int(round(prop*len(unames)))
  ## sampling with uniform dist and no replacement
  subsampled = np.random.choice(unames,size=n_samples,replace=False)

  output.write("\n".join(subsampled))
  print("Subsampled training set size: %d" % len(subsampled) )


  output_val = open(os.path.join(train_dir,"val.subsample_indexes"),'w')
  val_unames = [os.path.basename(fn) for fn in gb.glob(os.path.join(data_dir,"val","*.txt"))]
  
  prop = 0.4 if lang=="eus" else 1.0 # 0.1 for eus, 0.2 for quz
  n_samples = int(round(prop*len(val_unames)))
  ## sampling with uniform dist and no replacement
  subsampled = np.random.choice(val_unames,size=n_samples,replace=False)

  output_val.write("\n".join(subsampled))
  print("Subsampled validation set size: %d" % len(subsampled) )


