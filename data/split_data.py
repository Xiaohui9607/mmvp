import os
import glob

DATA_DIR = '../../data/CY101NPY'
OUT_DIR = './data/CY101NPY'

'''
Three kinds of splitting strategies
1. instance-based
2. object-based
3. category-based
'''

def split(root, ratio=(0.8, 0.1, 0.1), strategy="instance"):
    '''
    :param root: data_dir
    :param ratio: ratio [train, valid, test]
    :param strategy: splitting strategy
    :return: file list for train, valid, test
    '''
    if not strategy in ['instance', 'object', 'category']:
        raise KeyError('strategy should be instance | object | category')

    files = glob.glob(os.path.join(root, "*.npy"))
    pass

if __name__ == '__main__':
    split(DATA_DIR)