import yaml
import os
import pandas as pd
from boto3.session import Session
import cv2
import numpy as np
import ipyvolume as ipv
import joblib # joblib version: 0.9.4
import pickle
import errno

import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
SummaryWriter._PRINT_DEPRECATION_WARNINGS = False


class create_new_experiment:
    def __init__(self, cfg):
        self.cfg = cfg
    
        self.experiment_name = f"{cfg['train']['experimentName']}_from_e_{cfg['train']['startEpoch']:03d}" \
                          f"_to_{cfg['train']['endEpoch']:03d}_b_{cfg['train']['trainBatchSize']:03d}" \
                          f"_lr_{cfg['train']['lr']['init']:0.06f}"
                          
                          
        self.saveRoot = os.path.join(cfg['train']['resultDirectoryName'], cfg['train']['experimentFolder'],
                                        cfg['train']['experimentName'])
        
        self.sourefiles_path = os.path.join(self.saveRoot, 'source_file')
        self.model_checkpoints_path = os.path.join(self.saveRoot, 'model_checkpoints')
        
        
        self.logger_path = os.path.join(self.sourefiles_path, self.experiment_name + '.txt')
        
        self.initialize()
        self.create_logger()
        self.createSummaryWriter()
        #self.copy()
        
        
    @staticmethod  
    def copyanything(src, dst):
        try:
            shutil.copytree(src, dst)
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(src, dst)
            else:
                raise
        
  
        
        
    def initialize(self):
        if os.path.exists(self.saveRoot):
            del_dir = input(f"This directory exists. \nDo you want to 'REMOVE' [{self.saveRoot}] directory? (y/n) ")
            if del_dir == 'y': 
                shutil.rmtree(self.saveRoot , ignore_errors=True)
            else:
                assert del_dir == 'y', f"Program is terminated by typing:  ({del_dir})" 

        os.makedirs(self.saveRoot)
        os.makedirs(os.path.join(self.sourefiles_path,'script'))
        os.makedirs(self.model_checkpoints_path)
        
        #self.copyanything()
        
        
        
    def create_logger(self):
        if  self.cfg['options']['logger']['flag']:
            logging.basicConfig(filename= self.logger_path,
                            format='%(asctime)s %(message)s',
                            filemode='w')
                            
        # Creating an object
        self.logger = logging.getLogger()
        # Setting the threshold of logger to DEBUG
        self.logger.setLevel(logging.DEBUG)
        self.print_log(f"experiment name: {self.experiment_name}" ) 
        self.print_log(f"\nAll details can be found at: \n [{os.path.abspath(self.logger_path)}] \n {'='*20} \n")                    
                          
    def print_log(self, msg , log =True ,print_ = True):
        if print_:
            print(msg)
        if log:
            self.logger.info(f"[{os.path.basename(__file__)}] " + msg)

            
    def createSummaryWriter(self):
        self.writer = SummaryWriter(self.saveRoot)
            
    #def copy(self):
            
        #shutil.copyfile(__file__, os.path.join(self.sourefiles_path, os.path.basename(__file__)))
        ##print('__file__' ,__file__)
        # shutil.copyfile(os.path.abspath(yml_file),
                    # os.path.join(source_save_dir, os.path.basename(os.path.abspath(yml_file))))
        # self.copyanything(os.path.abspath('../lib'), os.path.join(source_save_dir, 'lib'))
        # self.copyanything(os.path.abspath('../lib2'), os.path.join(source_save_dir, 'lib2'))
        
        
                      



# class dataLogger:
    # def __init__(self, cfg ):
    
    
    
    
                                    
    
    
    # logger_path = os.path.join(source_save_dir, experiment_name + '.txt')
    


class ClogLossDataset_downloader:
    def __init__(self, config , online_data= True , draw_3d = False ,type='train'):
        self.cfg = config
        self.dataPath = config['dataset']['path']
        self.videoPath = os.path.join(config['dataset']['path'], 'video')
        self.online_data = online_data
        self.remove = config['dataset']['remove_donloaded_video']
        self.draw_3d = draw_3d
        self.saveDatasetDir  = config['dataset']['save_dir']
        self.download_fldr = 'downloded_data'
        self.size = config['dataset']['size']
        self.type = type
        
       # if not os.path.exists(self.download_fldr):
           # os.makedirs(self.download_fldr)


        if type=='train':
            metaData = os.path.join(self.dataPath ,'train_metadata.csv')
            metaData = pd.read_csv(metaData)

            label = os.path.join(self.dataPath ,'train_labels.csv')
            label = pd.read_csv(label)

            self.df_dataset = metaData
            self.df_dataset['stalled'] =label['stalled']
            self.df_dataset['vid_id'] = self.df_dataset.index
            with open(os.path.join(config['dataset']['path'], 'whole_train_dataset.pandas'), 'wb') as handel:
                pickle.dump(self.df_dataset, handel, protocol=pickle.HIGHEST_PROTOCOL)

        elif type=='test':
            metaData = os.path.join(self.dataPath, 'test_metadata.csv')
            metaData = pd.read_csv(metaData)
            self.df_dataset = metaData
            self.df_dataset['vid_id'] = self.df_dataset.index
            with open(os.path.join(config['dataset']['path'], 'whole_test_dataset.pandas'), 'wb') as handel:
                pickle.dump(self.df_dataset, handel, protocol=pickle.HIGHEST_PROTOCOL)

        
#         self.df_dataset = metaData[metaData['filename'].isin(df['filename'])]
#         self.df_dataset['stalled'] =label[label['filename'].isin(df['filename'])]['stalled']
#         self.df_dataset['vid_id'] = self.df_dataset.index
#         with open(os.path.join(config['dataset']['path'],'whole_train_dataset.pandas'),'wb') as handel:
#             pickle.dump(self.df_dataset , handel ,protocol=pickle.HIGHEST_PROTOCOL)
        
        
        if online_data:
            
            self.download_fldr = os.path.join(self.dataPath ,self.download_fldr )
            if not os.path.exists(f"./{self.saveDatasetDir}"):
                os.mkdir(self.saveDatasetDir)
            credentials_path = config['dataset']['credentials_path']
            with open (credentials_path , 'rb') as f:
                credentials = yaml.load(f ,Loader=yaml.FullLoader)
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
            df = pd.DataFrame([file for file in os.listdir(self.videoPath)  if file.split('.')[-1] == 'mp4'], columns=['filename'])
            self.df_dataset = self.df_dataset[metaData['filename'].isin(df['filename'])]
            self.df_dataset = self.df_dataset.reset_index(drop = True)
            
        # filter dataset
        print(f"Orginal Dataset >>>>>>> ", len(self.df_dataset))
        if type =='train':
            self.filter_dataset()
        # limit_data
        lim_min = self.cfg['dataset']['filter']['limit']['min']
        lim_max = self.cfg['dataset']['filter']['limit']['max']
        lim_flag = self.cfg['dataset']['filter']['limit']['flag']
        if lim_flag:
            self.df_dataset = self.df_dataset.iloc[lim_min :lim_max]
            
        with open(os.path.join(self.saveDatasetDir, f"{os.path.basename(self.saveDatasetDir)}.pandas"),'wb') as handel:
            pickle.dump(self.df_dataset , handel ,protocol=pickle.HIGHEST_PROTOCOL)
            
        self.number_of_objec = len(self.df_dataset)
        self.current_row=0
        
        
        
    def filter_dataset(self):
        filter_dict={}
        
        def apply_filter_each_row(row, filter_vector, col_name):
                return [True ,False][row[col_name] in filter_vector]
            
            
        
        try:
            filter_dict["tier1"] = self.cfg['dataset']['filter']['tier1']
        except:
            pass
        
        try:
            filter_dict["project_id"] = self.cfg['dataset']['filter']['project_id']
        except:
            pass
        
        try:
            filter_dict["stalled"] = self.cfg['dataset']['filter']['stalled']
        except:
            pass
        
        try:
            filter_dict["num_frames"] = self.cfg['dataset']['filter']['num_frames']
        except:
            pass
        
        try:
            filter_dict["crowd_score"] = self.cfg['dataset']['filter']['crowd_score']
        except:
            pass
        
        print(filter_dict)
        
        for key_ , filter_vec in filter_dict.items():
            
            if key_ in ['num_frames' ,'crowd_score']:
#                 print("--------------->",filter_vec , filter_vec[0])
                # filter more than max value
                filter_ = (self.df_dataset[self.df_dataset[key_] < filter_vec[0]].index.to_list()) 
                self.df_dataset = self.df_dataset.drop(filter_).reset_index(drop=True)

                
                # filter less than min value
                filter_ = self.df_dataset[self.df_dataset[key_] > filter_vec[1]].index.to_list()
                self.df_dataset = self.df_dataset.drop(filter_).reset_index(drop=True)
                print(f"Filtered  by[{key_}]  >>>>>>> ", len(self.df_dataset))
                
            else:
            
                filtred_indexex = self.df_dataset.apply(lambda row: apply_filter_each_row(row, filter_vec, key_),
                                                               axis=1)

                outliers = self.df_dataset[filtred_indexex].index.to_list()
                self.df_dataset = self.df_dataset.drop(outliers).reset_index(drop=True)

                print(f"Filtered  by[{key_}]  >>>>>>> ", len(self.df_dataset))
            

        
    
        
        
        
    def getFrame( self , vidcap , sec , image_name ):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames,image = vidcap.read()
        if(hasFrames):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image ,hasFrames

    @staticmethod
    def extract_location_area_from_highlighted_curve(image ,size):
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
        # plt.figure()
        # plt.imshow(np.hstack((image,vessels_image)))
        vessels_image = cv2.resize(vessels_image, (size[0], size[1]))
        vessels_image = cv2.cvtColor(vessels_image, cv2.COLOR_RGB2GRAY)

        #     area
        return vessels_image

    def get_specified_area(self , image):

            # convert to hsv to detect the outlined orange area
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            lower_red = np.array([100,120,150])
            upper_red = np.array([110,255,255])
            # create a mask
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            mask1 = cv2.dilate(mask1, None, iterations=2)
            mask_ind = np.where(mask1>0)
            xmin , xmax = min(mask_ind[1]) , max(mask_ind[1])
            ymin , ymax = min(mask_ind[0]) , max(mask_ind[0])
            # remove orange line from the image
            return mask1 ,(xmin , xmax , ymin , ymax)
    
    
    def filter_image(self, image ,mask1 ,area):
        xmin , xmax,ymin , ymax = area

        mask_ind = np.where(mask1>0)
        image[mask_ind ]=0,0,0
        # fill the area to skip the data outside of this area
        ret,mask1 = cv2.threshold(mask1,10,255,cv2.THRESH_BINARY_INV)
        contours,hierarchy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [ctr for ctr in contours if cv2.contourArea(ctr) < 5*(mask1.shape[0]*mask1.shape[1])/6]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         print(len(contours))
        cv2.drawContours(mask1, [contours[-1]], -1, (0, 0, 0), -1)
        # remove data out of the outlined area
        image[mask1>0] = (0,0,0)

    #     image =  cv2.rectangle(image , (xmin,ymin) ,(xmax,ymax),(255,255,255),4,4)
        image = image[ ymin:ymax , xmin:xmax ]
        image = cv2.resize(image ,(150,150))

#         image = image /255.
    #     image -= image.mean()
    #     image /= image.std()
    #     print(image.shape , xmin , xmax,ymin , ymax)
        return image
    
    @staticmethod
    def draw_tensor(tensor_img):

        ipv.figure()
#         ipv.volshow(tensor_img[...,0], level=[0.36, 0.55,1], opacity=[0.11,0.13, 0.13], level_width=0.05, data_min=0, data_max=1 ,lighting=True)
        ipv.volshow(tensor_img[...,0], level=[0.36, 0.17,0.36], opacity=[0.05,0.13, 0.10], level_width=0.05, data_min=0, data_max=1 ,lighting=True)
        
        ipv.view(-30, 45)
        ipv.show()
        
    def create_metadata(self,row):
        meta_data = {}
        meta_data['filename'] = row.filename
        meta_data['crowd_score'] = row.crowd_score
        meta_data['tier1'] = row.tier1
        meta_data['stalled'] = row.stalled
        meta_data['vid_id'] = row.vid_id
        meta_data['project_id'] = row.project_id
        meta_data['num_frames'] = row.num_frames
        return meta_data
    
    def __len__(self):
        return len(self.df_dataset)-1

    def __getitem__(self, index):
        row = self.df_dataset.iloc[index]
        if self.type == 'train':
            metadata = self.create_metadata(row)
        elif self.type == 'test':
            metadata = {}
            metadata['filename'] = row.filename
            metadata['vid_id'] = row.vid_id
            metadata['stalled'] = -1

        if self.online_data:
            vid_p = os.path.join(self.download_fldr, f"{row.filename}")
            self.bucket.download_file(f"{self.type}/{row.filename}", vid_p)
            vidcap = cv2.VideoCapture(vid_p)
        #
        else:
            vidcap = cv2.VideoCapture(os.path.join(self.videoPath, row.filename))
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        #         total_frames = config['dataset']['num_frames']
        # frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        Video_len = total_frames / fps
        from_sec = 0
        time_stamp = np.linspace(from_sec, Video_len, int(total_frames / 1.0))



        # for frame in range(int(total_frames)):
        # image , hasframe = self.getFrame(vidcap ,time_stamp[frame] , frame)

        # if hasframe:
        # if frame==0:
        # mask , area = self.get_specified_area(image)
        # image = self.filter_image(image , mask, area)
        # tensor_img.append(image)

        vessels_tensor = np.zeros([1, self.size[0], self.size[1]])
        for frame in range(int(total_frames)):
            image, hasframe = self.getFrame(vidcap, time_stamp[frame], frame)
            if hasframe:
                vessels_image = self.extract_location_area_from_highlighted_curve(image , self.size)
                vessels_tensor = np.append(vessels_tensor, vessels_image[np.newaxis, ...], axis=0)

        metadata['tensor_size'] = vessels_tensor.shape
        vidcap.release()
        if self.remove:
            os.remove(vid_p)
        vessels_tensor = np.array(list(vessels_tensor))
        #         print(tensor_img.shape)
        if self.draw_3d:
            self.draw_tensor(vessels_tensor)
        #         print(row)
        #         tensor_img = np.moveaxis(tensor_img,3,0)
        vessels_tensor = vessels_tensor.astype(np.uint8)
        joblib.dump([vessels_tensor, metadata], os.path.join(self.saveDatasetDir, f"{row.filename.split('.')[0]}.lzma"),
                    compress=('lzma', 6))
        #         print(os.path.join(self.saveDatasetDir ,f"{row.filename}"))

        return [vessels_tensor, metadata]
            
            
#         def __iter__(self):
#             return self

#         def __next__(self): # Python 2: def next(self)
#             row = self.df_dataset.iloc[self.current_row]
#             self.current_row +=1
#             if self.current_row < self.number_of_objec:
#                 return row
#             raise StopIteration
            
            
#video_downloder = ClogLossDataset_downloader(config  )
