from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
import pickle

app = FastAPI()

model = tf.keras.models.load_model('bi_lstm_final.h5', compile=False)
from_disk = pickle.load(open("vec_layer.pkl", "rb"))
vectorizer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
vectorizer.set_weights(from_disk['weights'])

class UserInput(BaseModel):
    prompt: str 


@app.get("/")
def index():
    return {"message": "Hello"}

@app.get('/{name}')
def get_name(name:str):
    return {'Hello': f'{name}'}

@app.post('/predict')
def predict_comment(data:UserInput):
    data = data.dict()
    comment = data['prompt']
    prompt_vec= vectorizer(comment)
    y_pred = model.predict(np.expand_dims(prompt_vec,0))
    y_pred = (y_pred > 0.5).astype(int)
    y_pred = y_pred.tolist()
    cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return {"classes":cols, "prediction": y_pred}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1" , port=8000)
                
#uvicorn toxicapi:app --reload