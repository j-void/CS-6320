import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################

  # imLoader = ImageLoader(root_dir=dir_name)

  image_files = glob.glob(os.path.join(dir_name, "**/*.jpg"), recursive=True)
  image_files.sort()

  scalar = StandardScaler()
  
  for ifile in image_files:
    img = Image.open(ifile).convert('L')
    img = np.asarray(img).astype(np.float32) / 255.
    scalar.partial_fit(img.reshape(-1, 1))
  
  mean, std = scalar.mean_, np.sqrt(scalar.var_)


  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
