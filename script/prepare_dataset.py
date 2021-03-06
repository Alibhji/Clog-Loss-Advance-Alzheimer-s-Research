# The Code written by Ali Babolhaveji @ 6/6/2020


import sys
package_path = '..'
if not package_path in sys.path:
    sys.path.append(package_path)

    from lib import ClogLossDataset_downloader
import yaml
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing  import  cpu_count

def call_a_video(index):
    global counter ,dataset_len

    _,meta=video_downloder[index]
    counter +=1
    print(f"{counter:6d} / {dataset_len} --> {(counter / dataset_len) * 100:07.5f} %  "
          f"[{meta['filename']}] [shape: {meta['tensor_size' ]}]")

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

dataset_type='test'
# config = './config.yml'
#config = '../script/configs/download/config_flowing.yml'
#config = '../script/configs/download/config_stall.yml'
config = '../script/configs/download/config_test_dataset.yml'

with open (config , 'rb') as f:
    config = yaml.load(f ,Loader=yaml.FullLoader)
    
    
cpu_count = config['dataset']['Multiprocessing_num_cores']
counter =0


# video_downloder = ClogLossDataset_downloader(config  , split='train')
video_downloder = ClogLossDataset_downloader(config  , type=dataset_type)

dataset_len = len(video_downloder)
print("The dataset len is :" ,dataset_len)
print("Number of cores    :" ,cpu_count)
indexes = range(dataset_len)

run_multi_proc()

#for i in tqdm(range(len(video_downloder))):
#    video_downloder[i]
