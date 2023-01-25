# The backend of deepstellar

## File structure

```
.
├── File
│   ├── cache
│   │   └── bert-base-uncased
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── checkpoint
│   │   └── toxic_ckpt_best.pth
│   └── profile
│       └── deep_stellar_p_20_s_39.profile
├── README.md
├── __init__.py
├── abstraction
│   ├── __init__.py
│   ├── profiling.py
│   └── utils.py
├── server_config.py
├── requirements.txt
├── server.py
└── simple_rnn.py
```

## usage
cache File is automatically generated, and files under bert-base-uncased will be downloaded when you first run `server.py`

For `ckpt_best.pth` and `deep_stellar_p_20_s_39.profile`, please download from: [google drive](https://drive.google.com/drive/folders/1Im4MkZNxfdd9onEI3hxyYZGYLENSnphs)

When all files are ready, you can set the server by using 

```bash
python3 server.py
```

to start the server. You can check more parameters in the `server`

To specify the path for model checkpoint or to set host and port of the server, you can either pass parameters as flags to python, or configure model_config.py


Example query:

```
curl http://127.0.0.1:5000/api/v1/predict\?input=I%20love%20you
```

For now we only use `GET` method.

