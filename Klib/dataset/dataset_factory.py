# The Code written by Ali Babolhaveji @ 6/9/2020
# this is designed based on Keras


import pandas as pd
import pickle
import os
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib


class DataSet():
    def __init__(self, config , datatype = 'train'):
        self.type = datatype
        self.config = config
        self.vid_path = os.path.join('..', config['dataset']['path'])


        with open(os.path.join(config['dataset']['path'] ,f'whole_{self.type}_dataset.pandas'), 'rb') as file:
            self.df_dataset = pickle.load(file)

        if self.type =='train':
            flow_folder = 'flowing_Tensors__'
            vids1 = os.listdir(os.path.join('..', config['dataset']['path'], flow_folder))
            vids1 = [vid.split('.')[0]+'.mp4' for vid in vids1]
            flowing_Tensors_pd = self.df_dataset.loc[(self.df_dataset['filename'].isin(vids1))]
            flowing_Tensors_pd['foldername'] = flow_folder

            vids2 = os.listdir(os.path.join('..', config['dataset']['path'], 'stall_Tensors'))
            vids2 = [vid.split('.')[0]+'.mp4' for vid in vids2]
            stall_Tensors_pd = self.df_dataset.loc[(self.df_dataset['filename'].isin(vids2))]
            stall_Tensors_pd['foldername'] ='stall_Tensors'

            if not os.path.exists(os.path.join(config['dataset']['path'], "temp_balanced_dataset_pd.pandas")):
                self.df_dataset = pd.DataFrame()
                #         self.flowing_Tensors_pd = self.flowing_Tensors_pd.iloc[:100]
                counter = 0
                len_stall = len(stall_Tensors_pd) - 1

                flowing_Tensors_pd = flowing_Tensors_pd.iloc[:2000]

                # balance data
                for rowt in tqdm(flowing_Tensors_pd.iterrows(), total=len(flowing_Tensors_pd)):
                    indx = rowt[0]
                    row = rowt[1]

                    self.df_dataset = self.df_dataset.append(row)
                    self.df_dataset = self.df_dataset.append(stall_Tensors_pd.iloc[counter])
                    counter = [0, counter + 1][counter < len_stall]

                self.df_dataset = self.df_dataset.reset_index(drop=True)

                with open(os.path.join(config['dataset']['path'], f"temp_balanced_dataset_pd.pandas"), 'wb') as handel:
                    pickle.dump(self.df_dataset, handel, protocol=pickle.HIGHEST_PROTOCOL)


            else:
                with open (os.path.join(config['dataset']['path'], f"temp_balanced_dataset_pd.pandas"),'rb') as handle:
                    self.df_dataset = pickle.load(handle)

            # self.df_dataset = self.df_dataset.iloc[:5000]
            self.df_dataset =self.df_dataset.drop(self.df_dataset.loc[self.df_dataset['filename'] == '684600.mp4'].index)
            #print("********************" , self.df_dataset.loc[self.df_dataset['filename'] == '684600.mp4'])

        elif self.type == 'test':
            self.df_dataset['foldername'] = 'test_dataset2'
            self.df_dataset['stalled'] = -1



        print(len(self.df_dataset))





        # self.df_dataset = self.df_dataset.loc[(self.df_dataset['filename'].isin(vids1) ) |
        #                                       (self.df_dataset['filename'].isin(vids2) ) ].reset_index(drop=True)



        print(self.df_dataset.columns)

    def split_train_test(self):
        return train_test_split(self.df_dataset, self.df_dataset.stalled, test_size=0.15, random_state=42)

    def preprocess_standard(self, x):
        return x / 255.

    #     def prepare_image(self, img, size, preprocessing_function, aug=False):
    #         img = scipy.misc.imresize(img, size)
    #         img = np.array(img).astype(np.float64)
    #         if aug: img = augment(img, np.random.randint(7))
    #         img = preprocessing_function(img)
    #         return img
    @staticmethod
    def getFrame(vidcap, sec, image_name):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if (hasFrames):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, hasFrames

    @staticmethod
    def extract_location_area_from_highlighted_curve(image, size, preprocessing_function):
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
        image[mask_ind] = 0, 0, 0
        # fill the area to skip the data outside of this area
        ret, mask1 = cv2.threshold(mask1, 10, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [ctr for ctr in contours if cv2.contourArea(ctr) < 5 * (mask1.shape[0] * mask1.shape[1]) / 6]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(mask1, [contours[-1]], -1, (0, 0, 0), -1)
        # remove data out of the outlined area
        image[mask1 > 0] = (0, 0, 0)

        image = image[ymin:ymax, xmin:xmax]

        mask2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, mask2 = cv2.threshold(mask2, 90, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours = [ctr for ctr in contours if cv2.contourArea(ctr) < 5*(mask1.shape[0]*mask1.shape[1])/6]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        vessels_image = np.zeros_like(image)

        areas = []
        for ctr in contours:
            if cv2.contourArea(ctr) > 50:
                cv2.drawContours(image, [ctr], -1, (255, 0, 0), -1)
                cv2.drawContours(vessels_image, [ctr], -1, (255, 255, 255), -1)

                xxmin, xxmax = min(ctr[:, :, 0])[0], max(ctr[:, :, 0])[0]
                yymin, yymax = min(ctr[:, :, 1])[0], max(ctr[:, :, 1])[0]
                #             image = cv2.rectangle(image , (xxmin ,yymin) ,(xxmax , yymax),(0,255,0),1,1)
                areas.append([xxmin, yymin, xxmax, yymax, cv2.contourArea(ctr)])
        #             print(xxmin ,yymin ,xxmax ,yymax)
        #         plt.figure()
        #         plt.imshow(np.hstack((image,vessels_image)))
        vessels_image = cv2.resize(vessels_image, (size[0], size[1]))
        vessels_image = cv2.cvtColor(vessels_image, cv2.COLOR_RGB2GRAY)
        vessels_image = preprocessing_function(vessels_image)
        #     area
        return vessels_image.astype(np.float32)

    def process_video(self, foldername, filename, size, preprocessing_function):

        path = os.path.join(self.vid_path, foldername, filename.split('.')[0])
        data__ = joblib.load(f"{path}.lzma")
        vessels_tensor = data__[0][:, :, :]
        #         print(filename)
        #         print(vessels_tensor.shape)

        #         total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        #         frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT )))
        #         fps = vidcap.get(cv2.CAP_PROP_FPS)
        #         Video_len = total_frames / fps
        #         from_sec = 0
        #         step  = 1.
        #         time_stamp = np.linspace(from_sec , Video_len , int(total_frames / step) )

        #         vessels_tensor = np.zeros([1,size[0],size[1]])

        #         for frame in range(int(total_frames)):
        #             image , hasframe = self.getFrame(vidcap ,time_stamp[frame] , frame)
        #             if hasframe:
        #                 vessels_image = self.extract_location_area_from_highlighted_curve(image ,size, preprocessing_function)
        #                 vessels_tensor =np.append(vessels_tensor ,vessels_image[np.newaxis,...], axis=0)




        if vessels_tensor.shape[0] < 100:
            vessels_tensor = np.append(vessels_tensor, np.zeros((100 - len(vessels_tensor), size[0], size[1])), axis=0)
        if vessels_tensor.shape[0] > 100:
            vessels_tensor = vessels_tensor[:100]



        return vessels_tensor[..., np.newaxis]

    def get_class_one_hot(self, tag, min_value=0):

        onHotTarget = np.ones((2), dtype=np.float32) * min_value

        onHotTarget[int(tag)] = 1
        return onHotTarget

    def data_generator(self, data, which_net='standard', size=(224, 224), batch_size=2):
        if which_net == 'resnet50':
            preprocessing_function = self.preprocess_input_resnet50
        elif which_net == 'densenet':
            preprocessing_function = self.preprocess_input_densenet
        elif which_net == 'inception':
            preprocessing_function = self.preprocess_input_inception
        elif which_net == 'vgg':
            preprocessing_function = self.preprocess_input_vgg16
        elif which_net == 'standard':
            preprocessing_function = self.preprocess_standard

        #         filename, tag = data.loc[0,['filename','stalled']].values
        #         processed_video = self.process_video(filename, size, preprocessing_function)
        #         return self.process_video(filename, size, preprocessing_function)

        while True:
            for start in range(0, len(data), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(data))
                data_batch = data[start:end]
                for foldername, filename, tag in data_batch.loc[:, ['foldername', 'filename', 'stalled']].values:
                    processed_video = self.process_video(foldername, filename, size, preprocessing_function)
                    x_batch.append(processed_video)
                    y_batch.append(self.get_class_one_hot(tag))
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                # print(x_batch.shape)
                if self.type =='train':
                    yield x_batch, y_batch
                else:
                    yield x_batch


#                 return x_batch, y_batch





