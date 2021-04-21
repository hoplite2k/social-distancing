import os
import pyrebase
from APIkeys import config

def cloud_upload(path):
    conf = config

    firebase = pyrebase.initialize_app(conf)
    storage = firebase.storage()
    storage.child(path).put(path)

    os.remove(path)



