# import model_manager
# from model_manager.matcher import GetRelevance
# from model_manager.matcher import GetSimilarity
# from model_manager.stance import DetectStance
import json

from model_manager.annotator import HollyHeuristic
from src.data_parser.utils import ProcessSingleEssay

if __name__ == '__main__':

    file = open("example_essay.txt", "r")
    # Read the entire contents of the file
    content = file.read()
    essays_sentences, topic = ProcessSingleEssay(content)
    g, score = HollyHeuristic(essays_sentences, topic)
    for a in g:
        print(a)


    # Specify the file path where you want to save the JSON file
    file_path1 = "result_final_sentences2.json"

    with open(file_path1, "w") as json_file:
        json.dump(g, json_file)

