# The Code written by Ali Babolhaveji @ 6/1/2020

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import ipyvolume as ipv
from boto3.session import Session
import os
import yaml


class ClogLossDataset(Dataset):
    def __init__(self, config, split='train', type='train', online_data=True):
        self.cfg = config
        self.dataPath = config['dataset']['path']
        self.videoPath = os.path.join(config['dataset']['path'], 'video')
        self.online_data = online_data

        metaData = os.path.join(self.dataPath, 'train_metadata.csv')
        metaData = pd.read_csv(metaData)

        label = os.path.join(self.dataPath, 'train_labels.csv')
        label = pd.read_csv(label)

        self.df_dataset = metaData
        self.df_dataset['stalled'] = label['stalled']

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


# train_Dataset = ClogLossDataset(config)