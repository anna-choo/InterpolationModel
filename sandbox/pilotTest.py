from ezc3d import c3d
import numpy as np
import pandas as pd
import collections, copy, os, itertools, random, sys
# from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.insert(1, SCRIPT_DIR)
from utils.GET_metafileDetails import metafile



class c3d_file:
    def __init__(self):
        self.mapping_anatomical_group_to_marker = {
            'head':[
            'RHEAD',
            'RTEMP',
            'LHEAD',
            'LTEMP'],

            'shoulder':[
            'RACR',
            'LACR'],

            'torso':[
            'STER',
            'XPRO',
            'C7',
            'T4',
            'T8',
            'T10'],
            

            'pelvis':[
            'LPSIS',
            'RPSIS',
            'RASIS',
            'LASIS',
            'LICR',
            'RICR'],

            'arms':[    
            'RHME',
            'RHLE',
            'RUA1',
            'RUA2',
            'RUA3',
            'RUA4',
            'LHME',
            'LHLE',
            'LUA1',
            'LUA2',
            'LUA3',
            'LUA4'],

            'wrist':[
            'LFA1',
            'LFA2',
            'LFA3',
            'LRSP',
            'LUSP',
            'RRSP',
            'RUSP',
            'RFA3',
            'RFA2',
            'RFA1'],

            'hand':[
            'LCAP',
            'LHMC1',
            'LHMC2',
            'LHMC3',
            'RCAP',
            'RHMC1',
            'RHMC2',
            'RHMC3'],

            'fingers':[
            'LTHUMB',
            'RTHUMB',
            'RMIDDLE',
            'LMIDDLE'],

            'thigh':[
            'LFLE',
            'LFME',
            'LTH1',
            'LTH2',
            'LTH3',
            'LTH4',
            'RFLE',
            'RFME',
            'RTH1',
            'RTH2',
            'RTH3',
            'RTH4'],

            'calves':[
            'LSK1',
            'LSK2',
            'LSK3',
            'LSK4',
            'LFAL',
            'LTAM',
            'RSK1',
            'RSK2',
            'RSK3',
            'RSK4',
            'RFAL',
            'RTAM'],

            'foot':[
            'RFCC',
            'RFMT5',
            'RFMT2',
            'RFMT1',
            'LFCC',
            'LFMT5',
            'LFMT2',
            'LFMT1']
            }
        # all tasks
        self.all_tasks = [ 
            "key_stand",
            "back", 
            "mouth", 
            "head", 
            "grasp", 
            "lateral", 
            "towel", 
            "step_down", 
            "step_up", 
            "kerb", 
            "balance", 
            "tug", 
            "10m", 
            "static", 
            "balance_static"]

        # different task needs different markerset; map accordingly by colour
        self.mapping_task_to_color = {
            'towel':'blue', 
            'grasp':'green', 
            'lateral':'green', 
            'mouth':'green', 
            'head':'green', 
            'back':'green', 
            'tug':'pink', 
            'key_stand':'pink', 
            'balance':'yellow', 
            'kerb':'yellow', 
            'step':'yellow', 
            '10m':'yellow',
            'step_down':'yellow',
            'step_up':'yellow',
            'balance_static':'yellow'}
        
        # Experimentation 1: travelling sales man (top down)
        self.mapping_color_to_anatomical_group = {
            'blue':['head', 'shoulder', 'torso', 'pelvis', 'arms', 'wrist', 'hand'],
            'green':['head', 'shoulder', 'torso', 'pelvis', 'arms', 'wrist', 'hand', 'fingers'],
            'pink':['head', 'shoulder', 'torso', 'pelvis', 'arms', 'wrist', 'hand', 'thigh', 'calves', 'foot'],
            'yellow':['head', 'shoulder', 'torso', 'pelvis', 'arms', 'wrist', 'thigh', 'calves', 'foot']
            }
        

    
    def read_c3d(self, c3d_path):
        # load c3d file
        self.c = c3d(c3d_path)

        # get all markers name in the file and their coordinates
        self.name_of_markers = self.c["parameters"]["POINT"]["DESCRIPTIONS"]["value"]
        self.coordinates_of_markers = self.c["data"]["points"][0:3,:,:]


        # function that will be used to capture frames of the missing markers
    def ranges(self, i):
        '''function to record continuous numbering'''
        for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

        
    def check_necessary_markers(self, task):
        '''Based on the task, check for the required markers: missing at any timeframe or missing entirely'''
        # map task to color to anatomical group to marker in the anatomical group
        anatomical_group = self.mapping_color_to_anatomical_group[self.mapping_task_to_color[task]]

        # rearrange the markers that are highly correlated together
        problematic_markers = collections.defaultdict()
        transformed_marker_sequence = []
        transformed_data =  None
        for group in anatomical_group:
            for marker in self.mapping_anatomical_group_to_marker[group]:
                
                # only consider those relevant markers
                if marker in self.name_of_markers:
                    idx =  self.name_of_markers.index(marker)

                    # check if marker has missing timeframes
                    missing_timeframes = list(self.ranges(np.argwhere(np.isnan(self.coordinates_of_markers[0,idx:idx+1,:]))[:,1]))
                    if len(missing_timeframes) != 0:
                        problematic_markers[marker] = missing_timeframes
                    
                    # start forming the sequence according to self.mapping_color_to_anatomical_group
                    if transformed_data is None:
                        transformed_data = self.coordinates_of_markers[:,idx:idx+1,:]
                    else:
                        transformed_data = np.concatenate((transformed_data, self.coordinates_of_markers[:,idx:idx+1,:]), axis = 1)
                    transformed_marker_sequence.append(marker)

                else:
                    # find out what are the missing markers, if any
                    problematic_markers[marker] = "ALL"
        return transformed_data, transformed_marker_sequence, problematic_markers

    def preprocess(self, data, minmax = False, znorm = False):
        missing_markers_mask =  np.isnan(data).any(axis = 1)
        
        # Invert the mask to get the valid timeframes
        valid_timeframes_mask = ~missing_markers_mask.any(axis=0)

        # Use the mask to drop timeframes with missing markers
        cleaned_data = data[:, :, valid_timeframes_mask]
        
        # minmax or z-score normalisation (standardisation) of the data
        manipulate_data = copy.copy(data)
        axes = ['x', 'y', 'z']

        # for inversion of scaling later
        scaling = collections.defaultdict()

        for i, axis in enumerate(axes):
            data_in_each_axis = manipulate_data[i:i+1,:,:]
            if znorm == True:
                # record down the standardisation factors for z-score normalisation 
                scaling["avg_" + axis] = np.mean(data_in_each_axis)
                scaling["sd_" + axis] = np.std(data_in_each_axis)
                # z-score normalise the data
                manipulate_data[i:i+1,:,:] = ((data_in_each_axis - scaling["avg_" + axis]) / scaling["sd_" + axis])
            

            if minmax == True:
                # record down the scaling factors for minmax
                scaling["min_" + axis] = np.min(data_in_each_axis)
                scaling["max_" + axis] = np.max(data_in_each_axis)
                # minmax the normalised data
                manipulate_data[i:i+1,:,:] = ((data_in_each_axis - scaling["min_" + axis]) / (scaling["max_" + axis] - scaling["min_" + axis]))
        
        if len(scaling) == 0:
            print("No scaling done")
        
        return cleaned_data, scaling
    #     self.cropped_transformed_data = None
    #     for timeframe in problematic_markers.keys():
    #         crop_start = timeframe[0]
    #         crop_end = timeframe[1]

    #         self.cropped_transformed_data = transformed_data[:,:,0:crop_start]



if __name__ == "__main__":

    train_dir = r"C:\Users\anna\Documents\gitHub\NTU\InterpolationModel\data\c3d\train"
    test_dir = r"C:\Users\anna\Documents\gitHub\NTU\InterpolationModel\data\c3d\test"

    # dir = r"C:\Users\Administrator\Documents\gitHub\NTU\InterpolationModel\data\train"
    # file = r"C:\Users\Administrator\Documents\gitHub\NTU\InterpolationModel\data\SN268\SN268_0027_10m_05_c3d.c3d"
    METAFILES_STORAGE_DIR = r"Z:\DataCollection\AbilityData\Normative\Processed\MetaFiles"
    train_subject_list = ["SN196", "SN200", "SN202", "SN268", "SN273", "SN279", "SN281", "SN292", "SN301"]
    test_subject_list = ["SN328"]

    # train_dir = r"C:\Users\Administrator\Documents\gitHub\NTU\InterpolationModel\data"
    # test_dir = r"C:\Users\Administrator\Documents\gitHub\NTU\InterpolationModel\data"

    # instantiate GET_metafiledetails
    metafile_fn = metafile()
    train_metadata = metafile_fn.gather_metafile(METAFILES_STORAGE_DIR, train_subject_list)
    train_metadata = train_metadata.reset_index()

    test_metadata = metafile_fn.gather_metafile(METAFILES_STORAGE_DIR, test_subject_list)
    test_metadata = test_metadata.reset_index()

    # instantiate c3d_file
    c3d_function = c3d_file()
    test_task = "10m"

    # get train dataset
    X = None
    for i in range(len(train_metadata)):
        file_name = train_metadata.loc[i, "renamed"] + ".c3d"
        task_name = train_metadata.loc[i, "task"]
        if task_name not in c3d_function.mapping_task_to_color.keys():
            continue
        if task_name != test_task:
            continue
        else:
            print(file_name, task_name)
            c3d_path = os.path.join(train_dir, file_name)
            c3d_function.read_c3d(c3d_path)
            temp_data, marker_sequence, problematic_markers = c3d_function.check_necessary_markers(task_name)
            train_data, scaling = c3d_function.preprocess(temp_data, minmax = True)
            if X is None:
                X = train_data
            else:
                X = pd.concat([X, train_data], axis = 0)
    
    for i in range(len(test_metadata)):
        file_name = test_metadata.loc[i, "renamed"] + ".c3d"
        task_name = test_metadata.loc[i, "task"]
        if task_name not in c3d_function.mapping_task_to_color.keys():
            continue
        if task_name != test_task:
            continue
        else:
            print(file_name, task_name)
            c3d_path = os.path.join(test_dir, file_name)
            c3d_function.read_c3d(c3d_path)
            temp_data, marker_sequence, problematic_markers = c3d_function.check_necessary_markers(task_name)
            test_data, scaling = c3d_function.preprocess(temp_data, minmax = True)
            if y is None:
                y = test_data
            else:
                y = pd.concat([X, test_data], axis = 0)


