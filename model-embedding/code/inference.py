import os
import time
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms


imgsz = 224

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def model_fn(model_dir):
    os.system('pip install smdebug')
    model = YOLO(os.path.join(model_dir, 'best.pt'))
    seq_model = torch.nn.Sequential(*list(model.model.model[:-1]))
    ClassifyModel = model.model.model[-1]
#     model.to(device)
#     model.eval()
    return seq_model, ClassifyModel


def get_embedding(im, model):
    seq_model, ClassifyModel = model
    im_coverted = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im_coverted)
    im = trans(im)
    im = im.unsqueeze_(dim=0)
    im = im.to(device)
    im = seq_model(im)
    if isinstance(im, list):
        x = torch.cat(im, 1)
    embedding = ClassifyModel.pool(ClassifyModel.conv(im)).flatten(1)
    embedding = embedding.cpu().tolist()[0]
    return embedding


def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        from six import BytesIO
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/x-npy':
        from io import BytesIO
        np_bytes = BytesIO(request_body)
        return np.load(np_bytes, allow_pickle=True)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body

    
def predict_fn(input_data, model):
#     print('[DEBUG] input_data type:', type(input_data), input_data.shape)
#     with torch.no_grad():
#         return model(input_data.to(device))
    pred = get_embedding(input_data, model)  # , size=imgsz
#     print('[DEBUG] pred:', pred, pred.xywhn)
#     pred.print()
    
    result = pred
            
#     print('[DEBUG] result:', result)
    
    return result


# def output_fn(prediction, content_type):
#     pass


if __name__ == '__main__':
    import cv2
    input_data = cv2.imread('../../minc-2500-tiny/test/brick/brick_001968.jpg')
    model = model_fn('../')
    result = predict_fn(input_data, model)
    print(result)