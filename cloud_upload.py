import os
import pyrebase
from APIkeys import config

def cloud_upload(path):
    conf = config

    firebase = pyrebase.initialize_app(conf)
    storage = firebase.storage() 

    local_path = path
    path_on_cloud = path
    storage.child(path_on_cloud).put(local_path)

    os.remove(path)



