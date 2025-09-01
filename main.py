#Python Core Modules
import os
import json
import random
import time

#Python External Modules
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

#nltk.download('punkt_tab')
#nltk.download('wordnet')  # Download packets for natural language took kit 


class chatbotmodel(nn.Module):


    def __init__(self,input,output):

        super(chatbotmodel,self).__init__()

        self.fully_connected_layer_01=nn.Linear(input,128)
        self.fully_connected_layer_02=nn.Linear(128,64)
        self.fully_connected_layer_03=nn.Linear(64,output)
        self.rectified_linear_unit= nn.ReLU(0.5) #breaks the linearity
        self.dropout_neurons=nn.Dropout(0.5) #drops 50 % of neurons

    def forward(self, x):

        x=self.rectified_linear_unit(self.fully_connected_layer_01(x))
        x=self.dropout_neurons(x)
        x=self.rectified_linear_unit(self.fully_connected_layer_02(x))
        x=self.dropout_neurons(x)
        x=self.fully_connected_layer_03(x)

        return x


class chatbotassistance:

    def __init__(self, intention_path, function_mappings=None):

        self.intention_path=intention_path

        self.model=None
        self.document=[]
        self.vocabulary=[]
        self.intentions=[]
        self.intentions_response={}
        self.function_mappings=function_mappings

        self.X=None # Matrix of pattern
        self.Y=None #Array of Greetings index

    @staticmethod
    def tokenise_and_lemmatize(text):

        lemmatizer=nltk.WordNetLemmatizer()

        words=nltk.word_tokenize(text) # Just take the words from the provided text
        words=[lemmatizer.lemmatize(word).lower() for word in words] # try to lemmatize means runs & run can combined

        return words


    def mapping_words_0_1(self,words):  # mapping the words and vocabulary to 1 , 0 , 1

        return [1 if word in words else 0 for word in self.vocabulary]
    

    def collecting_intention_information(self): # from the provided intention path 

        lemmatize=nltk.WordNetLemmatizer()

        if os.path.exists(self.intention_path):
            with open(self.intention_path,'r') as intentionfile:

                intention_data=json.load(intentionfile)

            for intents in intention_data['intents']:

                if intents['tag'] not in self.intentions:

                    self.intentions.append(intents['tag'])
                    self.intentions_response[intents['tag']]=intents['responses']

                for pattern in intents['patterns']:

                    pattern_words=self.tokenise_and_lemmatize(pattern) # return list if patten words runs , run will be [run, run]
                    self.vocabulary.extend(pattern_words)
                    self.document.append((intents['tag'],pattern_words))

                self.vocabulary=sorted(set(self.vocabulary))


    def prepare_data(self):

        bags=[]
        indecies=[]

        for d in self.document:

            words=d[1]
            bag=self.mapping_words_0_1(words)
            bags.append(bag)
            indecies.append(self.intentions.index(d[0]))

            # print(d[0],d[1])
            # print(bag)
            # print(bags)
            # print(indecies)
            # time.sleep(5)

        self.X=np.array(bags) #arrays of words value
        self.Y=np.array(indecies) #corresponding labels
        # print(self.X)
        # print(self.Y)

    def train_model(self,batch_size,lr,epoch):

        x_tensor=torch.tensor(self.X,dtype=torch.float)
        y_tensor=torch.tensor(self.Y,dtype=torch.long)

        dataset=TensorDataset(x_tensor,y_tensor)

        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        self.model=chatbotmodel(self.X.shape[1],len(self.intentions))

        loss_function=nn.CrossEntropyLoss()

        optimizer=optim.Adam(self.model.parameters(),lr=lr)

        for i in range(epoch):

            running_loss=0.0

            for batch_x,batch_y in dataloader:

                optimizer.zero_grad()
                output=self.model(batch_x)
                loss=loss_function(output,batch_y)
                loss.backward()
                optimizer.step()
                running_loss+=loss

            print(f'Epoch {i+1}: loss : {running_loss/len(dataloader):.4f}')


    def save_model(self,model_path,dimension_path):
        torch.save(self.model.state_dict(),model_path)

        with open(dimension_path,'w') as F:
            json.dump({'input_size': self.X.shape[1],'output_size':len(self.intentions)},F)


    def load_model(self,model_path,dimension_path):
        with open(dimension_path,'r') as F:
            dimension=json.load(F)

        self.model=chatbotmodel(dimension['input_size'],dimension['output_size'])
        self.model.load_state_dict(torch.load(model_path,weights_only=True))

    def process_message(self,input_message):
        words=self.tokenise_and_lemmatize(input_message)
        bag=self.mapping_words_0_1(words)

        bag_tensor= torch.tensor([bag],dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            prediction= self.model(bag_tensor)
        
        predicted_class_index=torch.argmax(prediction,dim=1).item()
        predicted_intent=self.intentions[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()
            
        if self.intentions_response[predicted_intent]:
            return random.choice(self.intentions_response[predicted_intent])
        else:
            return None

def get_stocks():
    stocks=['apple','tcs','google','nvidia','samsung']
    return random.sample(stocks,3)


if __name__ =="__main__":
    assistant=chatbotassistance('Intention.json',function_mappings={'stocks':get_stocks})
    assistant.collecting_intention_information()
    assistant.prepare_data()
    
    if os.path.exists('chatbot_model.pth') and os.path.exists('dimension.json'):
        assistant.load_model('chatbot_model.pth', 'dimension.json')
    else:
        assistant.train_model(batch_size=8, lr=0.001, epoch=100)
        assistant.save_model('chatbot_model.pth', 'dimension.json')


    while True:
        message =input("Enter the message \n")

        if message=='train':
                assistant.train_model(batch_size=8, lr=0.001, epoch=1000)
                assistant.save_model('chatbot_model.pth', 'dimension.json')
        print(assistant.process_message(message))



            




# obj1=chatbotassistance('Intention.json')
# obj1.collecting_intention_information()
# obj1.prepare_data()

            














