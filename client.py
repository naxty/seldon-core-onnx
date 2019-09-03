# pip install Pillow
from PIL import Image

# pip install numpy
import numpy as np

# pip install seldon-core
from seldon_core.seldon_client import SeldonClient

path_to_image = "images/smile.jpg"
image = Image.open(path_to_image).convert("L")
resized = image.resize((64, 64))
values = np.array(resized).reshape(1, 1, 64, 64)

"""
import json
json.dump({"data": {"ndarray": values.tolist()}}, open("payload.json","w"))
"""

# this is the ip from `minikube ip` and port from `kubectl get svc ambassador -o jsonpath='{.spec.ports[0].nodePort}'`
minikube_ambassador_endpoint = "192.168.99.100:30809"

deployment_name = "seldon-emotion"
namespace = "default"

sc = SeldonClient(
    gateway="ambassador",
    gateway_endpoint=minikube_ambassador_endpoint,
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace,
)

response = sc.predict(
    data=values, deployment_name=deployment_name, payload_type="ndarray"
)
print(response)
