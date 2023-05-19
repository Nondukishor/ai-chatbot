from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, Request, Response
)
from typing import List
from pydantic import BaseModel
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import nltk
import random
import warnings
warnings.filterwarnings('ignore')
import uvicorn


app = FastAPI()
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


class SocketManager:
    def __init__(self):
        self.active_connections: List[(WebSocket, str)] = []

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections.append((websocket, user))

    def disconnect(self, websocket: WebSocket, user: str):
        self.active_connections.remove((websocket, user))

    async def broadcast(self, data):
        for connection in self.active_connections:
            await connection[0].send_json(data)    

manager = SocketManager()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    print(text)
    ints = predict_class(text, model)
    return getResponse(ints, intents)

@app.get('/')
async def health():
    return {
        "message": "running....."
    }

@app.get("/api/current_user")
def get_user(request: Request):
    return request.cookies.get("X-Authorization")

class RegisterValidator(BaseModel):
    username: str

@app.post("/api/register")
def register_user(user: RegisterValidator, response: Response):
    print(user)
    response.set_cookie(key="X-Authorization", value=user.username, httponly=True)

@app.websocket("/api/chat")
async def chat(websocket: WebSocket):
    if sender := websocket.cookies.get("X-Authorization"):
        await manager.connect(websocket, sender)
        response = {
            "sender": sender,
            "message": "got connected"
        }
        await manager.broadcast(response)
        try:
            while True:
                data = await websocket.receive_json()
                response = chatbot_response(text=data['message'])
                await manager.broadcast(data=response)
        except WebSocketDisconnect:
            manager.disconnect(websocket, sender)
            response['message'] = "left"
            await manager.broadcast(response)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info", reload=True)





