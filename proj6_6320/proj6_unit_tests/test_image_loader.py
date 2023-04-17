from proj6_code.image_loader import ImageLoader
from proj6_code.data_transforms import get_fundamental_transforms

import numpy as np
import torch


def test_dataset_length():
  try:
    train_image_loader = ImageLoader(root_dir='data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

    test_image_loader = ImageLoader(root_dir='data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    train_image_loader = ImageLoader(root_dir='../data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

    test_image_loader = ImageLoader(root_dir='../data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

  assert train_image_loader.__len__() == 2985
  assert test_image_loader.__len__() == 1500


def test_unique_vals():
  try:
    train_image_loader = ImageLoader(root_dir='data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    train_image_loader = ImageLoader(root_dir='../data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

  item1 = train_image_loader.__getitem__(10)
  item2 = train_image_loader.__getitem__(25)

  assert not torch.allclose(item1[0], item2[0])


def test_class_values():
  try:
    test_image_loader = ImageLoader(root_dir='data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    test_image_loader = ImageLoader(root_dir='../data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

  class_labels = test_image_loader.class_dict

  expected_vals = {
      'OpenCountry': 0,
      'Industrial': 1,
      'Office': 2,
      'InsideCity': 3,
      'Kitchen': 4,
      'TallBuilding': 5,
      'Mountain': 6,
      'Forest': 7,
      'Store': 8,
      'LivingRoom': 9,
      'Street': 10,
      'Bedroom': 11,
      'Coast': 12,
      'Suburb': 13,
      'Highway': 14
  }

  assert len(class_labels) == 15
  assert set(class_labels.keys()) == set(expected_vals.keys())
  assert set(class_labels.values()) == set(expected_vals.values())


def test_load_img_from_path():
  try:
    test_image_loader = ImageLoader(root_dir='data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
    im_path = 'data/train/Bedroom/image_0003.jpg'
  except:
    test_image_loader = ImageLoader(root_dir='../data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

    im_path = '../data/train/Bedroom/image_0003.jpg'

  im_np = np.asarray(test_image_loader.load_img_from_path(im_path))

  try:
    expected_data = np.loadtxt('proj6_unit_tests/data/sample_inp.txt')
  except:
    expected_data = np.loadtxt('../proj6_unit_tests/data/sample_inp.txt')

  assert np.allclose(expected_data, im_np)


if __name__ == '__main__':
  test_load_img_from_path()
