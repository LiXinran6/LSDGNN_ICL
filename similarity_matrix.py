import numpy as np


emotion_positions = {
    "pleasure": (np.cos(np.pi / 20), np.sin(np.pi / 20)),
    "happiness": (np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
    "joy": (np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
    "pride": (np.cos(5 * np.pi / 20), np.sin(5 * np.pi / 20)),
    "elation": (np.cos(5 * np.pi / 20), np.sin(5 * np.pi / 20)),
    "excitement": (np.cos(7 * np.pi / 20), np.sin(7 * np.pi / 20)),
    "surprise": (np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
    "interest": (np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
    "anger": (-np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
    "irritation": (-np.cos(9 * np.pi / 20), np.sin(9 * np.pi / 20)),
    "hate": (-np.cos(7 * np.pi / 20), np.sin(7 * np.pi / 20)),
    "contempt": (-np.cos(5 * np.pi / 20), np.sin(5 * np.pi / 20)),
    "disgust": (-np.cos(3 * np.pi / 20), np.sin(3 * np.pi / 20)),
    "fear": (-np.cos(np.pi / 20), np.sin(np.pi / 20)),
    "boredom": (-0.5, 0),
    "disappointment": (-np.cos(np.pi / 20), -np.sin(np.pi / 20)),
    "frustration": (-np.cos(np.pi / 20), -np.sin(np.pi / 20)),
    "shame": (-np.cos(3 * np.pi / 20), -np.sin(3 * np.pi / 20)),
    "regret": (-np.cos(5 * np.pi / 20), -np.sin(5 * np.pi / 20)),
    "guilt": (-np.cos(7 * np.pi / 20), -np.sin(7 * np.pi / 20)),
    "sadness": (-np.cos(9 * np.pi / 20), -np.sin(9 * np.pi / 20)),
    "compassion": (np.cos(9 * np.pi / 20), -np.sin(9 * np.pi / 20)),
    "relief": (np.cos(7 * np.pi / 20), -np.sin(7 * np.pi / 20)),
    "admiration": (np.cos(5 * np.pi / 20), -np.sin(5 * np.pi / 20)),
    "love": (np.cos(3 * np.pi / 20), -np.sin(3 * np.pi / 20)),
    "contentment": (np.cos(np.pi / 20), -np.sin(np.pi / 20)),
    "neutral": (0, 0)
}

# 计算两个情感标签之间的余弦相似度  Compute the cosine similarity between two sentiment labels.
def cosine_similarity(p1, p2):
    dot_product = np.dot(p1, p2)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    if norm_p1 == 0 or norm_p2 == 0:
        return 0.0
   # print(norm_p2)
    return dot_product / (norm_p1 * norm_p2)

# 计算情感标签之间的相似度矩阵  Compute the similarity matrix between sentiment labels.
def compute_similarity_matrix(emotion_positions, n_dataset):
    emotions = list(emotion_positions.keys())
   # print(emotions)
    n = len(emotions)
    N = n_dataset  # 总情感标签数   Total number of sentiment labels.
  #  print(N)
    similarity_matrix = np.zeros((n, n))
    emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotions)}  # 标签到索引的映射   Mapping from labels to indices.
    # 获取 "neutral" 标签的索引 Get the index of the label "neutral".
    neutral_index = emotions.index("neutral")

    for i in range(n):
        for j in range(n):
            if i != j:
                v1 = emotion_positions[emotions[i]][0] 
                v2 = emotion_positions[emotions[j]][0]  
               # print(v1)
                p1 = emotion_positions[emotions[i]]
               # print(p1)
                p2 = emotion_positions[emotions[j]]
             #   print(v1)
                # 如果两个情感标签的价度极性相反，设相似度为0   If the valence polarities of two emotion labels are opposite, set their similarity to 0.
                if v1 * v2 < 0:
                    similarity_matrix[i][j] = 0
                elif v1 * v2 == 0:  # valence 极性为 0
                    similarity_matrix[i][j] = 1 / N  # 设置为 1/N  set to 1/N
                else:
                  
                   # print(v1)
                   # print(v2)
                   # print('-----')
                    similarity_matrix[i][j] = max(cosine_similarity(np.array(p1), np.array(p2)), 0)

    # 特殊处理 "neutral" 标签   Special handling for the "neutral" label.
    for i in range(n):
        if i != neutral_index:
            similarity_matrix[neutral_index][i] = 1 / N  # 与所有其他标签的相似度为 1/N The similarity between the "neutral" label and all other labels is set to 1/N.
            similarity_matrix[i][neutral_index] = 1 / N

    return similarity_matrix, emotion_to_index


def get_similarity_matrix(dataset):

    if dataset == 'IEMOCAP':
        n = 6
        similarity_matrix , emotion_to_index = compute_similarity_matrix(emotion_positions, n)
    else: 
        n = 7  
        similarity_matrix , emotion_to_index = compute_similarity_matrix(emotion_positions, n)

    #输出相似度矩阵
    #绝对值越接近1表示越相似，越接近0表示越不一样   The closer the absolute value is to 1, the more similar they are; the closer it is to 0, the more different they are.
   # print("Emotion Similarity Matrix:")
   # print(similarity_matrix)
   # print(emotion_to_index)
    return similarity_matrix,emotion_to_index
