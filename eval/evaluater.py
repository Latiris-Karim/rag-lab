#metrics I want to test
#context precision 
#context recall 

#step 1 get a dataset where you generate a question for each chunk answers then i have (question, chunk_id) as ground truth
#-> the questions are my queries

#step 2 context precision 
#for example, [A, B, C, D, E] are the retrieved chunks and only A, C are relevant to the question. Context precision = 2/5 = 0.4

#metric; context recall 
#for example, there are 10 relevant chunks in the dataset and only 2 of them were retrieved. Context recall = 2/10 = 0.2


class ContextPrecision:
    def __init__(self, relevant_chunk):
        self.relevant_chunk = relevant_chunk

class TestRun:
    ...

class Visualizer:
    ...
