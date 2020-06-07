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

def call_a_video(index):
    video_downloder[index]

def calculateParallel( threads= cpu_count):
    pool = ThreadPool(threads)
    results = pool.map(call_a_video, indexes)
    pool.close()
    pool.join()
    return results
    
def run_multi_proc() : 

    runprocess = calculateParallel( threads=cpu_count)
    print('\n - \n')
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
print(len(video_downloder))
indexes = range(len(video_downloder))

run_multi_proc()

#for i in tqdm(range(len(video_downloder))):
#    video_downloder[i]