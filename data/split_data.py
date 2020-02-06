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
categories = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc', 'cup',
              'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg', 'ball']


def split(root, ratio=(0.8, 0.1, 0.1), strategy="instance", K=5):
    '''
    :param root: data_dir
    :param ratio: ratio [train, valid, test]
    :param strategy: splitting strategy
    :param K: k_fold
    :return: k set of file list for train, valid, test
    '''
    if not strategy in ['instance', 'object', 'category']:
        raise KeyError('strategy should be instance | object | category')
    files = glob.glob(os.path.join(root, "*.npy"))
    if strategy == 'instance':
        execs = [[] for _ in range(5)]
        for file in files:
            for i in range(1,6):
                if 'exec_%s'%i in file:
                    execs[i-1].append(file)
    if strategy == 'object':
        pass

if __name__ == '__main__':
    split(DATA_DIR)