import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ast import literal_eval
import os
from tqdm import tqdm
import random
from multiprocessing import  Pool

random.seed(5749)
n_cores=8
plt.rcParams['figure.facecolor'] = 'black'

def draw_func(df):
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_id = row['id']
        symbol = row['key']
        if not os.path.exists('/Volumes/detext/' + symbol):
            os.makedirs('/Volumes/detext/' + symbol)
        coords = literal_eval(row['strokes'])
        for elem in coords:
            x,y,_ = np.array(elem).T
            plt.axis('off')
            plt.plot(x, -y, '-', color='white', lw=10)
            plt.scatter(x, -y, color='white', s=20)
        plt.savefig('/Volumes/detext/' + symbol + '/' + str(image_id)+'.png',facecolor='black', edgecolor='none', dpi=50)
        plt.clf()

if __name__=="__main__":
    df = pd.read_csv('../detexify-data/detexify-training-data/detexify.csv')

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    pool.map(draw_func, df_split)
    pool.close()
    pool.join()