import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, img_as_uint
from tqdm import tqdm_notebook, tqdm
from zipfile import ZipFile
import torch
import cv2
from DataLoader import ImagesetDataset, ImageSet
from DeepNetworks.HRNet import HRNet
from Evaluator import shift_cPSNR, cSSIM
from utils import getImageSetDirectories, readBaselineCPSNR, collateFunction


def get_sr_and_score(imset, model, aposterior_gt, num_frames, min_L=16):
    '''
    Super resolves an imset with a given model.
    Args:
        imset: imageset
        model: HRNet, pytorch model
        min_L: int, pad length
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
        scPSNR: float, shift cPSNR score
    '''

    if imset.__class__ is ImageSet:
        collator = collateFunction(num_frames, min_L=min_L)
        lrs, alphas, hrs, hr_maps, names = collator([imset])
    elif isinstance(imset, tuple):  # imset is a tuple of batches
        lrs, alphas, hrs, hr_maps, names = imset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)

    sr = model(lrs, alphas)[:, 0]
    sr = sr.detach().cpu().numpy()[0]


    mse_hrs = hrs.numpy()[0]
    mse_sr = sr
    mse_hr_map = hr_maps.numpy()[0]

    if len(sr.shape) == 2:
        mse_sr = mse_sr[None, ]
        mse_hrs = mse_hrs[None, ]
        mse_hr_map = mse_hr_map[None, ]

    n_clear = np.sum(mse_hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = mse_hrs -mse_sr
    bias = np.sum(diff * mse_hrs, axis=(1, 2)) / n_clear  # brightness bias
    cMSE = np.sum(np.square((diff - bias[:, None, None]) * mse_hr_map), axis=(1, 2)) / n_clear

#    print("HRS LEN: ", len(hrs))
#    print("HRS: ", hrs)

    if len(hrs) > 0:
        scPSNR = shift_cPSNR(sr=np.clip(sr, 0, 1),
                             hr=hrs.numpy()[0],
                             hr_map=hr_maps.numpy()[0])
        ssim = cSSIM(sr=np.clip(sr, 0, 1), hr=hrs.numpy()[0])
    else:
        scPSNR = None
        ssim = None

    #   print("APGT SHAPE: ", aposterior_gt.shape)
    #   print("APGT: ", aposterior_gt)

    if (str(type(aposterior_gt)) == "<class 'NoneType'>"):
        aposterior_ssim = 1.0
    else:
        aposterior_ssim = cSSIM(sr=np.clip(sr, 0, 1), hr=np.clip(aposterior_gt, 0, 1))

    return sr, scPSNR, ssim, aposterior_ssim, cMSE


def load_data(config_file_path, val_proportion=0.10, top_k=-1):
    '''
    Loads all the data for the ESA Kelvin competition (train, val, test, baseline)
    Args:
        config_file_path: str, paths of configuration file
        val_proportion: float, validation/train fraction
        top_k: int, number of low-resolution images to read. Default (top_k=-1) reads all low-res images, sorted by clearance.
    Returns:
        train_dataset: torch.Dataset
        val_dataset: torch.Dataset
        test_dataset: torch.Dataset
        baseline_cpsnrs: dict, shift cPSNR scores of the ESA baseline
    '''
    
    with open(config_file_path, "r") as read_file:
        config = json.load(read_file)

    data_directory = config["paths"]["prefix"]
    baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))

    train_set_directories = getImageSetDirectories(os.path.join(data_directory, "train"))
    test_set_directories = getImageSetDirectories(os.path.join(data_directory, "test"))

    # val_proportion = 0.10
    train_list, val_list = train_test_split(train_set_directories,
                                            test_size=val_proportion, random_state=1, shuffle=True)
    config["training"]["create_patches"] = False

    train_dataset = ImagesetDataset(imset_dir=train_list, config=config["training"], top_k=top_k)
    val_dataset = ImagesetDataset(imset_dir=val_list, config=config["training"], top_k=top_k)
    test_dataset = ImagesetDataset(imset_dir=test_set_directories, config=config["training"], top_k=top_k)
    return train_dataset, val_dataset, test_dataset, baseline_cpsnrs


def load_model(config, checkpoint_file):
    '''
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    '''
    
#     checkpoint_dir = config["paths"]["checkpoint_dir"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet(config["network"]).to(device)
    model.load_state_dict(torch.load(checkpoint_file))
    return model


def evaluate(model, train_dataset, val_dataset, test_dataset, min_L=16):
    '''
    Evaluates a pretrained model.
    Args:
        model: HRNet, a pytorch model
        train_dataset: torch.Dataset
        val_dataset: torch.Dataset
        test_dataset: torch.Dataset
        min_L: int, pad length
    Returns:
        scores: dict, results
        clerances: dict, clearance scores
        part: dict, data split (train, val or test)
    '''
    
    model.eval()
    scores = {}
    clerances = {}
    part = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for s, imset_dataset in [('train', train_dataset),
                             ('val', val_dataset),
                             ('test', test_dataset)]:

        if __IPYTHON__:
            tqdm = tqdm_notebook

        for imset in tqdm(imset_dataset):
            sr, scPSNR = get_sr_and_score(imset, model, min_L=min_L)
            scores[imset['name']] = scPSNR
            clerances[imset['name']] = imset['clearances']
            part[imset['name']] = s
    return scores, clerances, part


def custom_evaluate(model, train_dataset, val_dataset, test_dataset, num_frames, min_L=16):

    model.eval()
    scores = {}
    clerances = {}
    part = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for s, imset_dataset in [('train', train_dataset),
                             ('val', val_dataset),
                             ('test', test_dataset)]:

        if __IPYTHON__:
            tqdm = tqdm_notebook

        for imset in tqdm(imset_dataset):
            sr, scPSNR, ssim, aposterior_ssim = get_sr_and_score(imset, model, None, num_frames, min_L)
            #  imset, model, aposterior_gt, num_frames, min_L=16
            scores[imset['name']] = scPSNR
            clerances[imset['name']] = imset['clearances']
            part[imset['name']] = s
    return scores, clerances, part

def benchmark(baseline_cpsnrs, scores, part, clerances):
    '''
    Benchmark scores against ESA baseline.
    Args:
        baseline_cpsnrs: dict, shift cPSNR scores of the ESA baseline
        scores: dict, results
        part: dict, data split (train, val or test)
        clerances: dict, clearance scores
    Returns:
        results: pandas.Dataframe, results
    '''
    
    # TODO HR mask clearance
    results = pd.DataFrame({'ESA': baseline_cpsnrs,
                            'model': scores,
                            'clr': clerances,
                            'part': part, })
    results['score'] = results['ESA'] / results['model']
    results['mean_clr'] = results['clr'].map(np.mean)
    results['std_clr'] = results['clr'].map(np.std)
    return results


def generate_submission_file(model, imset_dataset, out='../submission'):
    '''
    USAGE: generate_submission_file [path to testfolder] [name of the submission folder]
    EXAMPLE: generate_submission_file data submission
    '''

    print('generating solutions: ', end='', flush='True')
    os.makedirs(out, exist_ok=True)
    if __IPYTHON__:
        tqdm = tqdm_notebook

    for imset in tqdm(imset_dataset):
        folder = imset['name']
        sr, _ = get_sr_and_score(imset, model)
        sr = img_as_uint(sr)

        # normalize and safe resulting image in temporary folder (complains on low contrast if not suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(out, folder + '.png'), sr)
            print('*', end='', flush='True')

    print('\narchiving: ')
    sub_archive = out + '/submission.zip'  # name of submission archive
    zf = ZipFile(sub_archive, mode='w')
    try:
        for img in os.listdir(out):
            if not img.startswith('imgset'):  # ignore the .zip-file itself
                continue
            zf.write(os.path.join(out, img), arcname=img)
            print('*', end='', flush='True')
    finally:
        zf.close()
    print('\ndone. The submission-file is found at {}. Bye!'.format(sub_archive))

    
    

    
class Model(object):
    
    def __init__(self, config):
        self.config = config
        
    def load_checkpoint(self, checkpoint_file):
        self.model = load_model(self.config, checkpoint_file)
        
    def __call__(self, imset, aposterior_gt, num_frames, custom_min_L = 16):
        sr, scPSNR, gt_SSIM, aposterior_SSIM, cMSE = get_sr_and_score(imset, self.model, aposterior_gt, num_frames, min_L= custom_min_L)#self.config['training']['min_L'])
        return sr, scPSNR, gt_SSIM, aposterior_SSIM, cMSE
    
    def evaluate(self, train_dataset, val_dataset, test_dataset, baseline_cpsnrs):                
        scores, clearance, part = evaluate(self.model, train_dataset, val_dataset, test_dataset, 
                                           min_L=self.config['training']['min_L'])

        results = benchmark(baseline_cpsnrs, scores, part, clearance)
        return results

    def custom_evaluate(self, train_dataset, val_dataset, test_dataset, baseline_cpsnrs, num_frames, min_L):
        scores, clearance, part = custom_evaluate(self.model, train_dataset, val_dataset, test_dataset, num_frames, min_L)

        results = benchmark(baseline_cpsnrs, scores, part, clearance)
        return results
    
    def generate_submission_file(self, imset_dataset, out='../submission'):
        generate_submission_file(self.model, imset_dataset, out='../submission')
