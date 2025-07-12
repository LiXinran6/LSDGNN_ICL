import numpy as np
import similarity_matrix

class Dialog:
    def __init__(self, utterances, labels, speakers, features, dataset):
        self.utterances = utterances
        self.labels = labels
        self.speakers = speakers
        self.features = features
        self.dataset = dataset
        self.numberofemotionshifts = 0
        self.numberofspeakers = 0
        self.numberofutterances = 0
        self.difficulty = 0
        self.emotion_variance = 0  # 情感变化度量
        self.emotion_shift_weighted = 0 #加权后的情感变化
        self.cc()

    def __getitem__(self, item):
        if item == 'utterances':
            return self.utterances
        elif item == 'labels':
            return self.labels
        elif item == 'speakers':
            return self.speakers
        elif item == 'features':
            return self.features
    #measure the difficulty of a dialog
    def cc(self):
        # 情感数字到文字的映射字典
      # print(self.dataset)
        if self.dataset == 'MELD':
            emotion_map = { -1: 'null', 0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}
        else:
            emotion_map = { -1: 'null', 0:'excitement', 1: 'neutral', 2:'frustration', 3:'sadness', 4:'happiness', 5:'anger'}
     #   print(emotion_map)
        self.numberofutterances = len(self.utterances)
        speaker_emo = {}
        for i in range(0, len(self.labels)):
            if (self.speakers[i] in speaker_emo):
                speaker_emo[self.speakers[i]].append(emotion_map[self.labels[i]])
            else:
                speaker_emo[self.speakers[i]] = [emotion_map[self.labels[i]]]

        # 获取情感相似度矩阵
        matrix, emotion_to_index = similarity_matrix.get_similarity_matrix(self.dataset)
       # print(matrix)
        k = 1
        b = 0.4
        for key in speaker_emo:
          #  prev_emo = None
            for i in range(0, len(speaker_emo[key]) - 1):
                current_emo = speaker_emo[key][i]
                next_emo = speaker_emo[key][i + 1]
                if current_emo != next_emo and current_emo != 'null' and next_emo != 'null':
                    self.numberofemotionshifts += 1
                    current_emo_index = emotion_to_index[current_emo]
                    next_emo_index = emotion_to_index[next_emo]
                    #线性缩放
                    #当k为正数时，similarity_score越小说明差距越大，越大说明差距越小,侧重于差距小的情感
                    #当k为负数时，反之，侧重于差距大的情感
                    similarity_score = abs(matrix[current_emo_index][next_emo_index]) * k + b
                    self.emotion_shift_weighted += similarity_score
                    
        #print(speaker_emo[key]) 
        '''
        for key in speaker_emo:
            # Convert labels to indices
            emotions = speaker_emo[key]
            self.emotion_variance += np.std(emotions)  # 计算每个发言人的情感方差
       '''
       # print(self.numberofemotionshifts)
       # print(self.emotion_shift_weighted)
       # print('---------')
        self.numberofspeakers = len(set(self.speakers))
        self.difficulty = (self.emotion_shift_weighted + self.numberofspeakers ) / (self.numberofutterances + self.numberofspeakers) 
       # self.difficulty = (self.numberofemotionshifts + self.numberofspeakers ) / (self.numberofutterances + self.numberofspeakers) 