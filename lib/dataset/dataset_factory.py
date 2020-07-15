# The Code written by Ali Babolhaveji @ 6/7/2020

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import ipyvolume as ipv
from boto3.session import Session
import os
import yaml
import joblib
import pickle
from tqdm import tqdm


from sklearn.model_selection import KFold


def genrate_K_folds(dataset_pd , K=2 , Fold =0 ,random_state=None ,shuffle=False):


    kf = KFold(n_splits=K, random_state=random_state, shuffle=shuffle)
    assert Fold < kf.n_splits , f"Fold number should be between [0 and {K-1}] but it is {Fold} "
    for num,(train, val) in enumerate(kf.split(dataset_pd)):
#         print(f"Fold-{num}",'train: %s, val: %s' % (train, val))
        if Fold == num:
            return train, val
    


class ClogLossDataset(Dataset):
    def __init__(self, config, split='train', type='train', online_data=True):
        self.cfg = config
        self.dataPath = config['dataset']['path']
        self.videoPath = os.path.join(config['dataset']['path'], 'video')
        self.online_data = online_data
        if type == 'train':
            metaData = os.path.join(self.dataPath, 'train_metadata.csv')
            metaData = pd.read_csv(metaData)

            label = os.path.join(self.dataPath, 'train_labels.csv')
            label = pd.read_csv(label)

            self.df_dataset = metaData
            self.df_dataset['stalled'] = label['stalled']

        elif type == 'test':
            metaData = os.path.join(self.dataPath, 'test_metadata.csv')
            metaData = pd.read_csv(metaData)
            self.df_dataset = metaData


        #         self.df_dataset = metaData[metaData['filename'].isin(df['filename'])]
        #         self.df_dataset['stalled'] =label[label['filename'].isin(df['filename'])]['stalled']
        self.df_dataset['vid_id'] = self.df_dataset.index

        if True:
            self.download_fldr = 'downloded_data'
            self.download_fldr = os.path.join(self.dataPath, self.download_fldr)
            if not os.path.exists(f"./{self.download_fldr}"):
                os.mkdir(self.download_fldr)
            credentials_path = config['dataset']['credentials_path']
            with open(credentials_path, 'rb') as f:
                credentials = yaml.load(f, Loader=yaml.FullLoader)
            #                 print(credentials)

            ACCESS_KEY = credentials['ACCESS_KEY']
            SECRET_KEY = credentials['SECRET_KEY']

            session = Session(aws_access_key_id=ACCESS_KEY,
                              aws_secret_access_key=SECRET_KEY)
            s3 = session.resource('s3')
            self.bucket = s3.Bucket('drivendata-competition-clog-loss')
        #             self.df_dataset = self.df_dataset[self.df_dataset['num_frames'] > 200]
        #             train_Dataset.df_dataset[train_Dataset.df_dataset['tier1']== True]

        #             self.df_dataset = self.df_dataset[self.df_dataset['stalled']==0]

        #             for s3_file in your_bucket.objects.all():
        #                 print(s3_file.key) # prints the contents of bucket

        else:
            df = pd.DataFrame([file for file in os.listdir(self.videoPath) if file.split('.')[-1] == 'mp4'],
                              columns=['filename'])
            self.df_dataset = self.df_dataset[metaData['filename'].isin(df['filename'])]
            self.df_dataset = self.df_dataset.reset_index(drop=True)



    #         self.df_dataset['num_frames'].plot.hist()
    #         self.df_dataset['stalled'] = label[label['filename'].isin(df['filename'])]

    #         print((label.iloc[570501]))
    #         print((self.df_dataset))

    def getFrame(self, vidcap, sec, image_name):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if (hasFrames):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, hasFrames

    def get_specified_area(self, image):

        # convert to hsv to detect the outlined orange area
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([100, 120, 150])
        upper_red = np.array([110, 255, 255])
        # create a mask
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask1 = cv2.dilate(mask1, None, iterations=2)
        mask_ind = np.where(mask1 > 0)
        xmin, xmax = min(mask_ind[1]), max(mask_ind[1])
        ymin, ymax = min(mask_ind[0]), max(mask_ind[0])
        # remove orange line from the image

        return mask1, (xmin, xmax, ymin, ymax)

    def filter_image(self, image, mask1, area):
        xmin, xmax, ymin, ymax = area

        mask_ind = np.where(mask1 > 0)
        image[mask_ind] = 0, 0, 0
        # fill the area to skip the data outside of this area
        ret, mask1 = cv2.threshold(mask1, 10, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [ctr for ctr in contours if cv2.contourArea(ctr) < 5 * (mask1.shape[0] * mask1.shape[1]) / 6]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #         print(len(contours))
        cv2.drawContours(mask1, [contours[-1]], -1, (0, 0, 0), -1)
        # remove data out of the outlined area
        image[mask1 > 0] = (0, 0, 0)

        #     image =  cv2.rectangle(image , (xmin,ymin) ,(xmax,ymax),(255,255,255),4,4)
        image = image[ymin:ymax, xmin:xmax]
        image = cv2.resize(image, (150, 150))
        image = image / 255.
        #     image -= image.mean()
        #     image /= image.std()
        #     print(image.shape , xmin , xmax,ymin , ymax)
        return image

    @staticmethod
    def draw_tensor(tensor_img):

        ipv.figure()
        ipv.volshow(tensor_img[..., 0], level=[0.36, 0.55, 1], opacity=[0.11, 0.13, 0.13], level_width=0.05, data_min=0,
                    data_max=1, lighting=True)
        ipv.view(-30, 45)
        ipv.show()

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, index):
        row = self.df_dataset.iloc[index]
        #         print(row)
        if self.online_data:
            vid_p = os.path.join(self.download_fldr, f"{row.filename}")
            self.bucket.download_file(f"train/{row.filename}", vid_p)
            vidcap = cv2.VideoCapture(vid_p)
        #
        else:
            vidcap = cv2.VideoCapture(os.path.join(self.videoPath, row.filename))
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        #         total_frames = config['dataset']['num_frames']
        frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        Video_len = total_frames / fps
        from_sec = 0
        time_stamp = np.linspace(from_sec, Video_len, int(total_frames / 1.0))

        tensor_img = []

        for frame in range(int(total_frames)):
            image, hasframe = self.getFrame(vidcap, time_stamp[frame], frame)

            if hasframe:
                if frame == 0:
                    mask, area = self.get_specified_area(image)
                image = self.filter_image(image, mask, area)
                tensor_img.append(image)

            if frame >= 199:
                break

        if len(tensor_img) < 200:
            for kk in range(200 - len(tensor_img)):
                tensor_img.append(list(np.zeros([150, 150, 3])))

        # print(len(tensor_img))
        vidcap.release()
        os.remove(vid_p)
        tensor_img = np.array(list(tensor_img) ,dtype=np.float32)
        # print(tensor_img.shape)

        # self.draw_tensor(tensor_img)
        #         print(row)
        tensor_img = np.moveaxis(tensor_img, 3, 0)

        return tensor_img







class ClogLossDataset_from_compressed_data(Dataset):
    def __init__(self, config, split='train', type='train' , draw_3d = False , fold=0):
        self.cfg = config
        self.dataPath = config['dataset']['path']
        self.draw_3d = draw_3d
        self.fold = fold
        self.split =split
        with open (os.path.join(self.dataPath , 'flowing_Tensors','flowing_Tensors.pandas'),'rb') as handle:
            self.flowing_Tensors_pd = pickle.load(handle)
        self.flowing_Tensors_pd['folder_name'] = np.tile('flowing_Tensors',len(self.flowing_Tensors_pd))
        
        with open (os.path.join(self.dataPath , 'stall_Tensors','stall_Tensors.pandas'),'rb') as handle:
            self.stall_Tensors_pd = pickle.load(handle)
        self.stall_Tensors_pd['folder_name'] = np.tile('stall_Tensors',len(self.stall_Tensors_pd))
        
        if not os.path.exists(os.path.join(self.dataPath, "temp_balanced_dataset_pd.pandas")):
            self.df_dataset = pd.DataFrame()
    #         self.flowing_Tensors_pd = self.flowing_Tensors_pd.iloc[:100]
            counter = 0   
            len_stall = len(self.stall_Tensors_pd)-1


            self.flowing_Tensors_pd = self.flowing_Tensors_pd.iloc[:2000]


            # balance data
            for  rowt in tqdm(self.flowing_Tensors_pd.iterrows() , total = len(self.flowing_Tensors_pd)):
                indx=rowt[0]
                row =rowt[1]

                self.df_dataset = self.df_dataset.append(row)
                self.df_dataset = self.df_dataset.append(self.stall_Tensors_pd.iloc[counter])
                counter = [0 ,counter+1][counter <len_stall] 

            self.df_dataset = self.df_dataset.reset_index(drop=True)

            with open(os.path.join(self.dataPath, f"temp_balanced_dataset_pd.pandas"),'wb') as handel:
                pickle.dump(self.df_dataset , handel ,protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            with open (os.path.join(self.dataPath, f"temp_balanced_dataset_pd.pandas"),'rb') as handle:
                self.df_dataset = pickle.load(handle)

        # self.df_dataset = self.df_dataset.iloc[:5000]
        self.df_dataset =self.df_dataset.drop(self.df_dataset.loc[self.df_dataset['filename'] == '684600.mp4'].index)
        #print("********************" , self.df_dataset.loc[self.df_dataset['filename'] == '684600.mp4'])


        train, val = genrate_K_folds(self.df_dataset, K=self.cfg['dataset']['K'], Fold=self.fold)
        
        if self.split == 'train':
            self.df_dataset = self.df_dataset.iloc[train]

        elif self.split == 'val':
            self.df_dataset = self.df_dataset.iloc[val]
            
    #         print(self.df_dataset )
        
    @staticmethod
    def draw_tensor(tensor_img):

        ipv.figure()
        ipv.volshow(tensor_img[..., 0], level=[0.36, 0.55, 1], opacity=[0.11, 0.13, 0.13], level_width=0.05, data_min=0,
                    data_max=1, lighting=True)
        ipv.view(-30, 45)
        ipv.show()

            
            
    def __len__(self):
        return len(self.df_dataset)
    
    def __getitem__(self, index):
        row = self.df_dataset.iloc[index]
        file= os.path.join(self.dataPath,row.folder_name  , row.filename.split('.')[0]+'.lzma')
        data = joblib.load(file)
        tensor_img = data[0]
        meta = data [1]
        
        #print(tensor_img.mean())
#         tensor_img = tensor_img - tensor_img.mean()
        tensor_img = tensor_img/255.0
        
        if tensor_img.shape[0]<80:
            #print(tensor_img.shape[0])
            tensor_img = np.append(tensor_img , np.zeros((80 - len(tensor_img),150, 150, 3)),axis=0)
        # if tensor_img.shape[0] > 199:
        tensor_img = tensor_img[:80]
        
#         print(tensor_img.mean())
        if self.draw_3d:
            self.draw_tensor(tensor_img)
        meta['tier1']= str(meta['tier1'])

        onHotTarget = np.ones((2), dtype=np.float32 )* -1
        onHotTarget[meta['stalled']] = 1
        meta['target'] = onHotTarget

        tensor_img = np.moveaxis(tensor_img ,3,0)
        tensor_img = tensor_img.astype(np.float32)

        return tensor_img , meta

    
