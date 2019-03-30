import os
import glob
import h5py
import cv2
import numpy as np



def normalize(img, mean_value):
    return img - mean_value

def get_mean(root_path, file_names):
    all_mean_value = [0., 0., 0.]
    for i, file_name in enumerate(file_names):
        img = cv2.imread(os.path.join(root_path, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2,0,1])/255.
        mean_value = img.mean(-1).mean(-1)
        print('img {}, mean value {}'.format(i, mean_value))
        all_mean_value += mean_value
    print('all {} imgs, mean value {}'.format(len(file_names), all_mean_value/len(file_names)))
    return all_mean_value/len(file_names)

def get_img(file_path, mean_value):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2,0,1])/255.
    img = normalize(img, np.array(mean_value)[:, np.newaxis, np.newaxis])
    return img

def generate_h5(hr_path, lr_path, scaling_factor, h5_name):
    hr_file_names = glob.glob(os.path.join(hr_path, '*.png'))
    lr_file_name_tem = '{}' + scaling_factor + '.png'
    # mean_value = get_mean(root_path, file_names)
    mean_value = [0.44845608, 0.43749626, 0.40452776]
    h5f = h5py.File(os.path.join(hr_path, h5_name), mode='w')
    for i, hr_file_name in enumerate(hr_file_names):
        hr = get_img(os.path.join(hr_path, hr_file_name), mean_value)
        lr_file_name = lr_file_name_tem.format(hr_file_name.split('/')[-1].split('.')[0])
        print('here')
        print(os.path.join(lr_path, scaling_factor, lr_file_name))
        lr = get_img(os.path.join(lr_path, scaling_factor.upper(), lr_file_name), mean_value)
        print(hr.shape, hr.max(), hr.mean(), hr.min())
        print(lr.shape, lr.max(), lr.mean(), lr.min())
        exit()


if __name__=="__main__":
    hr_path = '/data0/langzhiqiang/DIV2K/DIV2K_train_HR/'
    lr_path = '/data0/langzhiqiang/DIV2K/DIV2K_train_LR_bicubic/'
    scaling_factor = 'x4'
    h5_name = 'train.h5'
    generate_h5(hr_path, lr_path, scaling_factor, h5_name)