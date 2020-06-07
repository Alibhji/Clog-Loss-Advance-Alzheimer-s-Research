# The Code written by Ali Babolhaveji @ 6/6/2020


import sys
package_path = '..'
if not package_path in sys.path:
    sys.path.append(package_path)

from lib import ClogLossDataset_downloader
import yaml
from tqdm import tqdm
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing  import  cpu_count
cpu_count =4
counter =0

def call_a_video(index):
    global counter ,dataset_len
    counter +=1
    video_downloder[index]
    print(f"{counter:6d} / {dataset_len} --> {(counter / dataset_len) * 100:02.5f} %")
    return index

def calculateParallel( threads= cpu_count):
    pool = ThreadPool(threads)
    results = pool.map(call_a_video, indexes)
    #print(results)
    pool.close()
    pool.join()
    return results
    
def run_multi_proc() : 

    runprocess = calculateParallel( threads=cpu_count)
    #print('\n runprocess \n')
    # j = tqdm(squaredNumbers)
    # j.set_description(f'Creating pandas dataset...')
    # for n in j:
        # tt = pd.DataFrame.from_dict(n, orient='index')
        # Dataset_pd = Dataset_pd.append(tt ,ignore_index=True )
        ##print(pd.DataFrame(n))
    # return Dataset_pd
    
    




config = './config.yml'
with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)

video_downloder = ClogLossDataset_downloader(config )

dataset_len = len(video_downloder)
print(dataset_len)
indexes = range(dataset_len)

run_multi_proc()

#for i in tqdm(range(len(video_downloder))):
#    video_downloder[i]