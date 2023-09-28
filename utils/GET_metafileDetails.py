import os
import pandas as pd


######## NOT YET CONSIDERED
# random rows/columns that appear out of nowhere
class metafile:
    def __init__(self, old_tasks = False):

        # Do not take old tasks
        if old_tasks ==  False:
            self.task_dict = {
                    "unilateral" : ["key_stand", 
                                    "back", 
                                    "mouth", 
                                    "head", 
                                    "grasp", 
                                    "lateral", 
                                    "towel"],

                    "bilateral" : ["step_down", 
                                   "step_up", 
                                   "kerb", 
                                   "balance", 
                                   "tug", 
                                   "10m", 
                                   "static", 
                                   "balance_static", 
                                   "step"]
                    }
            
        # Take old tasks
        elif old_tasks == True:
            self.task_dict = {
                    "unilateral" : ["key_stand", 
                                    "back", 
                                    "mouth", 
                                    "head", 
                                    "grasp", 
                                    "lateral", 
                                    "towel", 
                                    "bb", 
                                    "lateral_cube", 
                                    "key_sit"],
                    "bilateral" : ["step_down", 
                                    "step_up", 
                                    "kerb", 
                                    "balance", 
                                    "tug", 
                                    "10m", 
                                    "static", 
                                    "balance_static", 
                                    "step",

                                    "static_norm", 
                                    "functional", 
                                    "functional_ll",
                                    "functional_ul",
                                    "static_wand_a",
                                    "static_wand_b",
                                    "static_wand_c",
                                    "bend",
                                    "step2",
                                    "c3d_rom",
                                    "more_test"
                                   ]
                    }
        self.standard_header = ['file id', 'task', 'hand/leg', 'trial #', 'trial fail #', "sensor fail #", 'remarks']
        # self.metafile = self.process(METAFILE_PATH)
        # print(self.metafile)


    def check_storage(self, METAFILES_STORAGE_DIR, subject_list):
        metafile_notFound =  []
        metafile_found = []
        for subject in subject_list:
            # check if meta file exist
            metafile = os.path.join(METAFILES_STORAGE_DIR, "meta files - " + str(subject) + ".xlsx")
            if os.path.exists(metafile):
                print("metafile found for", subject)
                metafile_found.append(subject)
            else:
                # do not update if meta file does not exist, go to next subject
                print("metafile not found for", subject)
                # print(metafile)
                metafile_notFound.append(subject)
        return metafile_found, metafile_notFound



    def read(self, METAFILE_PATH,  metafileStandardHeader = "file id"):
        # read metafile
        metafile_details = pd.read_excel(METAFILE_PATH)

        # get subject 
        subject = metafile_details.columns[1]

        # use standard header to determine which row does the data recording starts 
        # this is needed because sometimes comments are included in the meta file, we want to skip those rows
        column_list = []
        for column in metafile_details.columns:
            column_list.append(metafile_details.index[metafile_details.loc[:,column] == metafileStandardHeader].tolist())
        for i in range(len(column_list)):
            if len(column_list[i]) != 0:
                index = column_list[i][0]
                column = i
                break
        metafile_details.columns = metafile_details.iloc[index]
        metafile_details.drop([i for i in range(index + 1)], inplace = True)
        metafile_details = metafile_details.reset_index(drop = True)

        # get rid of whitespace, if any in header and task
        metafile_details.columns =  [str(i).strip() for i in metafile_details.columns]
        metafile_details["task"] =  metafile_details["task"].str.strip()
        metafile_details["file id"] = metafile_details["file id"].astype(str)
        metafile_details["file id"] =  metafile_details["file id"].str.strip()


        # exclude any empty rows in the recording (especially when empty rows are recorded at the end)
        blank_space = [str(i).isspace() for i in metafile_details["file id"]]
        if True in blank_space:
            metafile_details.drop([str(i).isspace() for i in metafile_details["file id"]].index(True), inplace = True)
        else:
            pass
        metafile_details.columns.name = None

        # only take successful trials (consider trial fail and sensor fail)
        ## taken into consideration variation of "all", e.g. "All", "all", "ALL", etc
        ## change integers in "trial fail #" column to str
        ## taken into consideration if there are blankspace in "trial fail #"

        success_trials = metafile_details[metafile_details['trial fail #'].astype(str).str.strip().str.upper() != "ALL"]
        # exclude old trials
        ## taken into consideration variation of task, e.g. "Static", "static", etc
        success_trials = success_trials[success_trials['task'].str.lower().isin(self.task_dict["unilateral"] + self.task_dict["bilateral"])]
        success_trials["task"] = success_trials["task"].apply(lambda x: x.strip().lower())


        # e.g. Naming convention: [subject]_[video]_[t ask]_[repetition]
        # Unilateral: SN001_0008_key_stand_R01 (R,L recorded for unilateral tasks but not bilateral tasks)
        # Bilateral: SN345_0033_kerb_02

        # to record the counting process for all tasks
        counter = {}
        video_name_list = []
        for i,k in success_trials.iterrows():
    
            # getting the [repetition] part
            if success_trials.loc[i, "task"] in self.task_dict['unilateral']:
                ## taken into consideration if there are blankspace in file_id,  task
                counter_name = success_trials.loc[i, "task"] + success_trials.loc[i, "hand/leg"]
            elif success_trials.loc[i, "task"] in self.task_dict['bilateral']:
                ## taken into consideration if there are blankspace in file_id,  task
                counter_name = success_trials.loc[i, "task"]

            # update counter 
            if counter_name not in counter:
                counter[counter_name] = 1
            else:
                counter[counter_name] += 1

            # naming according to naming convention
            if success_trials.loc[i, "task"] in self.task_dict['unilateral']:
                ## taken into consideration if there are blankspace in file_id,  task, hand/leg
                video_name = subject + '_' + str(success_trials.loc[i, "file id"]).strip().zfill(4) + '_' + str(success_trials.loc[i, "task"]) + '_' + str(success_trials.loc[i,  "hand/leg"]).strip() + str(counter[counter_name]).zfill(2)
            elif success_trials.loc[i, "task"] in self.task_dict['bilateral']:
                ## taken into consideration if there are blankspace in file_id,  task
                counter_name = success_trials.loc[i, "task"]
                video_name = subject + '_' + str(success_trials.loc[i, "file id"]).strip().zfill(4) + '_' + str(success_trials.loc[i, "task"]) + '_' + str(counter[counter_name]).zfill(2)
            video_name_list.append(video_name)
        raw_video = []

        # only keep data that are within the standard header
        success_trials.columns = [str(x).lower() for x in success_trials.columns]
        keepIndex = [list(success_trials.columns).index(i) for i in self.standard_header]
        success_trials = success_trials.iloc[:,keepIndex]

        # store old and new name into df
        for id in success_trials["file id"]:
            raw = subject + '_' + str(id).strip().zfill(4)
            raw_video.append(raw)
        success_trials['renamed'] = video_name_list
        success_trials['original'] = raw_video
        success_trials = success_trials.reset_index()
        return success_trials

    def gather_metafile(self, subject_list, METAFILES_STORAGE_DIR):
        '''gather all the metafile details of all subjects'''
        # check is metafile exist
        metafile_found, metafile_notFound = self.check_storage(METAFILES_STORAGE_DIR, subject_list)
        all_metafile_details = None
        for subject in metafile_found:
            METAFILE_PATH = os.path.join(METAFILES_STORAGE_DIR, subject)
            metafile_details = self.read(METAFILE_PATH)
            if all_metafile_details is None:
                all_metafile_details = metafile_details
            else:
                all_metafile_details = pd.concat([all_metafile_details, metafile_details], axis = 0)
        
        return all_metafile_details

if __name__ == "__main__":
    METAFILES_STORAGE_DIR = r"Z:\DataCollection\AbilityData\Normative\Processed\MetaFiles"
    metafile = os.path.join(METAFILES_STORAGE_DIR, "meta files - " + str("SN054") + ".xlsx") 
    metafile_details = metafile.read(metafile, old_tasks=True)

    print(metafile_details)