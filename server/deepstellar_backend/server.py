# Given a sentence, return the prediction and trace

from transformers import BertTokenizerFast
import argparse
import torch
import joblib
import numpy as np
import os
import warnings
import json
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
from flask_cors import CORS, cross_origin


from simple_rnn import SimpleGRU, SimpleGRUTranslation, SimpleGRUMultiClassification, SimpleLSTM
from abstraction.profiling import DeepStellar
import server_config

token_model_name = "bert-base-uncased"


def get_model_arch(model_class):
    """
    Given model class as model.__class__, return corresponding architecture.
    """
    if model_class == "SimpleGRU":
        return SimpleGRU
    elif model_class == "SimpleGRUTranslation":
        return SimpleGRUTranslation
    elif model_class == "SimpleGRUMultiClassification":
        return SimpleGRUMultiClassification
    elif model_class == "SimpleLSTM":
        return SimpleLSTM
    else:
        raise Exception("Model not found")


class DeepstellarPredictor():
    
    def __init__(self, 
                 device, 
                 tokenizer_cache, 
                 arch, 
                 checkpoint, 
                 deepstellar_path):
        """
        Deepstellar model server version.
        Args
        ---
        - device: str cpu or cuda.
        - tokenizer_cache: where to store the tokenizer
        - arch: architecture of the model
        - checkpoint: the path for model checkpoint
        - deepstellar_path: the path for deepstellar model
        """
        self.device = device
        if not os.path.exists(tokenizer_cache):
            os.mkdir(tokenizer_cache)
        tokenizer_file = os.path.join(tokenizer_cache, token_model_name)
        if not os.path.exists(tokenizer_file):
            tokenizer = BertTokenizerFast.from_pretrained(token_model_name)
            tokenizer.save_pretrained(tokenizer_file)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_file)
        self.tokenizer = tokenizer
        arch = get_model_arch(arch)
        if device == "cpu":
            ckpt = torch.load(checkpoint, map_location=torch.device(self.device))
        else:
            ckpt = torch.load(checkpoint)
        model = arch(*ckpt['model_args'])
        model.load_state_dict(ckpt['model'])
        model.eval()
        model.to(self.device)
        self.model = model
        deep_stellar_model = joblib.load(deepstellar_path)
        self.deep_stellar_model = deep_stellar_model
    
    def pre_process(self, sentence: str):
        token_input = self.tokenizer.encode_plus(sentence, add_special_tokens=False,return_token_type_ids=True,
                                                 return_attention_mask=True, padding='max_length',
                                                 max_length=256, truncation=True)
        input_tensor = torch.LongTensor(token_input['input_ids']).to(self.device)
        mask_ = np.array(token_input['attention_mask'])
        if len(input_tensor.shape) < 3:
            input_tensor = input_tensor.unsqueeze(0)
        processed_input = {"input_tensor": input_tensor, "mask":mask_}
        return processed_input 
    
    def predict(self, processed_input: dict):
        input_tensor = processed_input["input_tensor"]
        text_ = np.array(self.tokenizer.convert_ids_to_tokens(input_tensor[0].cpu().numpy()))
        mask_ = processed_input["mask"]
        hidden_states, pred_tensor = self.model.profile(input_tensor)
        hidden_states = hidden_states.cpu().numpy()[0][mask_ == 1.]
        input_pca = self.deep_stellar_model.pca.do_reduction([hidden_states])
        input_trace = self.deep_stellar_model.get_trace(input_pca)
        prediction = pred_tensor[0].cpu().numpy() >= 0.5
        prediction = prediction.astype(int)
        pred = prediction[mask_ == 1.][-1][0]
        seq_label = prediction[mask_ == 1.].reshape(-1)
        text_ = text_[mask_ == 1.].tolist()
        result = {"input_trace":input_trace[0].tolist(), "prediction":int(pred), 'seq_label':seq_label.tolist(),
        'text':text_}
        return result
  
if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # Attention. Ignore warning (Most from sklearn)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="The path for the model",
                        default=server_config.model_checkpoint)
    parser.add_argument('--deepstellar', type=str, help="The path for deepstellar model",
                        default=server_config.deepstellar_path)
    parser.add_argument('--gpu', action='store_true', help="Set this flag to use GPU")
    parser.add_argument('--arch', 
                        choices=["SimpleGRU", "SimpleGRUTranslation", "SimpleGRUMultiClassification", "SimpleLSTM"],
                        help="The model architecture",
                        default=server_config.model_arch
                        )
    parser.add_argument('--tcache', type=str, help="The folder for tokenizer cache",
                        default=server_config.tokenizer_cache)
    parser.add_argument('--port', type=int, default=server_config.PORT, help="The server port")
    parser.add_argument('--host', type=str, default=server_config.HOST, help="The server host")
    
    args = parser.parse_args()
    device = 'cuda' if args.gpu else 'cpu'
    ds_predictor = DeepstellarPredictor(device, args.tcache, args.arch, args.checkpoint, args.deepstellar)
    app = Flask("deepstellar")
    
    @app.route("/api/v1/predict/", methods=['GET'])
    @cross_origin()
    def get():
        #parser = reqparse.RequestParser()  # initialize
        #parser.add_argument('input', required=True)
        sentence = request.args.get('input')
        processed_input = ds_predictor.pre_process(sentence)
        result = ds_predictor.predict(processed_input)
        return {'result': result}, 200  # return data and 200 OK code
    
    CORS(app, origins=server_config.allowed_origins, resources=r'/api/*')
    api = Api(app)
    #api.add_resource(PredictServer, '/api/v1/predict')  
    app.run(host=args.host, port=args.port)
    
    
    """
    curl http://127.0.0.1:5000/api/v1/predict\?input=I%20love%20you
    """