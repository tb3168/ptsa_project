import os
import random
import scipy
from scipy.stats import gengamma
import matplotlib.pyplot as plt
import numpy as np
import datetime

class_name_index = np.array(['', 'flood', 'blip', 'pulse-chain', 'box', 'snow', 'misc'])

def flood(duration=2 * 60., a=None, c=None, peak=1., power=1., noise=0.):
    duration = int(duration)
    if a is None:
        a = np.exp(np.random.uniform(0.6, 2))
    if c is None:
        c = 1-np.exp(np.random.uniform(0.6, 2))
    xi = np.linspace(
        gengamma.ppf(0.001, a, c),
        gengamma.ppf(0.99, a, c), round(duration))
    y = gengamma.pdf(xi, a, c) ** power
    # y = np.concatenate([np.zeros(1), y])
    y = y * peak / y.max()
    y = y + np.random.randn(len(y)) / 10 * noise
    return y


def blip(duration=0., peak=1.0, start=0.0, end=0.0, jump=None):
    return np.array([
        start, peak, 
        *([peak * jump] if jump is not None else []), 
        *([end] if end is not None else [])])


def box(duration=30., peak=1.0, lead=0.0, tail=0.0, noise=0.0):
    duration = max(int(duration), 2)
    top = np.ones(int(duration)) * peak + np.random.randn(int(duration)) / 10 * noise
    return np.concatenate([
        np.array([lead]) if lead is not None else np.array([]), 
        top, 
        np.array([tail]) if tail is not None else np.array([]),
    ])


def tiered_box(duration=60., split=0.5, jump=None, peak=1., **kw):
    duration = int(duration)
    if split is None:
        split = np.random.uniform(1/duration, 1-1/duration)
    if jump is None:
        jump = np.random.randn()/4 + np.random.choice([-0.1, 0.1]) + 1

    return np.concatenate([
        box(int(round(duration*split)), peak=peak, tail=None, **kw), 
        box(int(round(duration*(1-split))), peak=peak*jump, lead=None, **kw)])


def zeros(duration=10.):
    return np.zeros(int(duration))


def pulse_chain(duration=60., sparsity=0.2, box_noise=0.1, **kw):
    def sample(duration):
        return _sample_func([
            ('box', lambda d: box(d, noise=box_noise)),
            ('blip', lambda d: blip(d, end=np.random.choice([None, 0], p=[0.7, 0.3]))),
            ('box', lambda d: tiered_box(d, noise=box_noise)),
            ('', lambda d: zeros(d//2)),
        ], [0.4, 0.3, 0.1, sparsity], duration)
    return chain(int(duration), sample, **kw)



def chain(duration: float, func, peak=1., density=5, noise=0., size_var=0., **kw):
    duration = int(duration)
    ys = []
    labels = []
    progress = []
    count = 0
    while count < duration:
        # sample duration
        dur = duration - count
        dur2 = min(duration // density, dur)
        dur = np.random.choice([
            np.random.uniform(0, dur2), 
            np.random.uniform(dur2, dur*0.9)
        ], p=[1-size_var, size_var])
        dur = int(dur)

        # sample
        y, label = func(duration, **kw)

        # scaling noise
        y = y * (1 + np.random.randn() / 10 * noise) * peak
        count += len(y)
        ys.append(y)
        labels.append(np.array([label]*len(y)))
        progress.append(np.linspace(0, 1, len(y)))

    idxs = np.arange(len(ys))
    np.random.shuffle(idxs)
    y = np.concatenate([ys[i] for i in idxs])
    labels = np.concatenate([labels[i] for i in idxs])
    progress = np.concatenate([progress[i] for i in idxs])
    return y, labels, progress

def _sample_func(funcs, p, duration):
    p = np.asarray(p)
    i = np.random.choice(range(len(funcs)), p=p/p.sum())
    name, func = funcs[i]
    return func(duration), name

def discretization(y, scale):
    return y - y % scale


def main(duration=1*60*24, num_fakes=30, **kw):
    import tqdm
    import pandas as pd

    def sample(duration, noise=0, noise_peak=1, flood_peak=1):
        noise_peak = np.random.uniform(10, 800)
        flood_peak = np.random.uniform(30, 300)
        noise = 0.1

        funcs, p = random.choice([
            ([
                ('box', lambda d: box(
                    np.random.uniform(5, 2*60), noise=noise,
                    peak=noise_peak)),
                ('blip', lambda d: blip(
                    jump=random.choice([None, 0.8, 1.2]), 
                    peak=noise_peak)),
                ('box', lambda d: tiered_box(
                    np.random.uniform(10, 60), noise=noise,
                    peak=noise_peak)),
                ('pulse-chain', lambda d: pulse_chain(
                    np.random.uniform(60, 4*60), 
                    size_var=0.4, noise=0.2, box_noise=noise,
                    peak=noise_peak)[0]),
                
            ], [1, 1, 1, 1]),
            ([
                ('flood', lambda d: flood(
                    np.random.uniform(60, 4*60), noise=noise,
                    peak=flood_peak)),
            ], [1]),
            ([
                ('', lambda d: zeros(np.random.uniform(2, 30))),
            ], [1]),
        ])
        y, label = _sample_func(funcs, p, duration)
        return np.concatenate([y,  np.zeros(4)]), label

        # return _sample_func([
        #     ('box', lambda d: box(d)),
        #     ('blip', lambda d: blip(d)),
        #     ('box', lambda d: tiered_box(d)),
        #     ('flood', lambda d: flood(d*2)),
        #     ('pulse-chain', lambda d: pulse_chain(d*2)[0]),
        #     ('', lambda d: zeros(d//2)),
        # ], [0.4, 0.3, 0.1, sparsity], duration)



    os.makedirs('data/fake', exist_ok=True)

    for i in tqdm.tqdm(range(num_fakes)):
        dep_id = f'fake_{i}'
        y, label, progress = chain(duration, sample, **kw)
        lookup = {k: i for i, k in enumerate(class_name_index)}

        class_idx = pd.Series([lookup[k] for k in label])
        df = pd.DataFrame({
            'time':  pd.date_range(None, datetime.datetime.now(), len(y), '1min'),
            'deployment_id': dep_id,
            'depth_filt_mm': y,
            'depth_proc_mm': y,
            'label': label,
            'full_multi_class': class_idx,
            'binary_class': (class_idx == lookup['flood']).astype(int),
            'flood_progress': progress,
        })
        df.loc[df.label != 'flood', 'flood_progress'] = np.nan

        print(df.label.value_counts())
        print(df.binary_class.value_counts())
        print(df.depth_filt_mm.describe())
        df.to_csv(f'data/fake/{dep_id}.csv')
        

    



    

#if __name__ == '__main__':
 #   import fire
  #  fire.Fire(main)