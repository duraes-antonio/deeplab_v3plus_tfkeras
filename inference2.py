import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import glob
import matplotlib
from PIL import Image

model_dir = "./test_SH_cut"
traindata_dir = '../../data/train_data'
validdata_dir = '../../data'

valid_names = ["valid_4-09", "valid_4-09"]

deeplabv3plus_dir="./src"
sys.path.append(deeplabv3plus_dir)
image_size = (512,512)

gpu_options = tf.compat.v1.GPUOptions(visible_device_list="2", allow_growth=False)
config = tf.compat.v1.ConfigProto(gpu_options = gpu_options)
tf.compat.v1.enable_eager_execution(config=config)

from data_utils import make_xy_from_data_paths, convert_y_to_image_array
from data_gen import DataGenerator
from label import Label
from metrics import IoU
from loss import make_overwrap_crossentropy
from tensorflow.keras.utils import get_custom_objects
label_file_path = os.path.join(traindata_dir, 'label.csv')
label = Label(label_file_path)
get_custom_objects()["IoU"] = IoU
get_custom_objects()["overwrap_crossentropy"] = make_overwrap_crossentropy(label.n_labels)

out_dir = os.path.join(model_dir,"figure")
model = keras.models.load_model(os.path.join(model_dir,'best_model.h5'))
preprocess = keras.applications.xception.preprocess_input

last_activation = model.layers[-1].name

train_x_paths = glob.glob(os.path.join(traindata_dir,'*.png'))
train_x_paths.sort()
image_names = [os.path.basename(train_x_paths[i]).split('.')[0] for i in range(len(train_x_paths))]
train_y_paths=[]
for i, image_name in enumerate(image_names):
    p = os.path.join(traindata_dir, image_name+'.json')
    if os.path.exists(p):
        train_y_paths.append(p)
    else:
        train_y_paths.append(None)



valid_x, valid_y = make_xy_from_data_paths(train_x_paths,
                                           train_y_paths,
                                           image_size,
                                           label,
                                           "polygon",
                                           resize_or_crop="crop")

pred = model.predict(preprocess(valid_x), batch_size=8)

y_pred = convert_y_to_image_array(pred, label, threshold=0.5, activation=last_activation)
y_true = convert_y_to_image_array(valid_y, label, activation=last_activation)
out_dir_train = os.path.join(out_dir, "train")
os.makedirs(out_dir_train,exist_ok = True)
matplotlib.use('Agg')
for i in range(len(y_pred)):
    if last_activation == "softmax":
        img = Image.fromarray(y_pred[i])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_pred_seg.png"))
        img = Image.fromarray(y_true[i])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_true_seg.png"))

        img = Image.fromarray(valid_x[i,:,:,:])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_x.png"))

        y_mask = y_pred[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        #img = Image.fromarray(y_mask*valid_x[i,:,:,:]/255)
        img = Image.fromarray((y_mask*valid_x[i,:,:,:]).astype(np.uint8))
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_pred_x_seg.png"))

        y_mask = y_true[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        img = Image.fromarray((y_mask*valid_x[i,:,:,:]).astype(np.uint8))
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_true_x_seg.png"))
    elif last_activation == "sigmoid":
        img = Image.fromarray(valid_x[i,:,:,:])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_x.png"))
        for j in range(label.n_labels):
            label_name =label.name[j]

            img = Image.fromarray(y_pred[i][j])
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_pred_seg.png"))
            img = Image.fromarray(y_true[i][j])
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_true_seg.png"))

            y_mask = y_pred[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            #img = Image.fromarray(y_mask*valid_x[i,:,:,:]/255)
            img = Image.fromarray((y_mask*valid_x[i,:,:,:]).astype(np.uint8))
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_pred_x_seg.png"))

            y_mask = y_true[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            img = Image.fromarray((y_mask*valid_x[i,:,:,:]).astype(np.uint8))
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_true_x_seg.png"))

for valid_name in valid_names:
    valid_data_dir = os.path.join(validdata_dir, valid_name)

    valid_x_paths = glob.glob(os.path.join(valid_data_dir,'*.png'))
    valid_x_paths.sort()

    tar = range(len(valid_x_paths))

    from data_utils import inference_large_img
    from PIL import Image
    from tqdm import tqdm

    mode = "max_confidence"
    out_dir_valid = os.path.join(out_dir, "valid_" + valid_name)
    os.makedirs(out_dir_valid, exist_ok=True)
    for i in tqdm(tar):
        x_img, seg_img = inference_large_img(valid_x_paths[i],
                                             model,
                                             preprocess,
                                             label,
                                             mode=mode,
                                             threshold=0.5)

        if last_activation == "softmax":
            img = Image.fromarray(seg_img)
            img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_seg.png"))

            img = Image.fromarray(x_img)
            img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_x.png"))

            y_mask = seg_img.copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            y_mask[black_pix,:] = [1.0,1.0,1.0]
            img = Image.fromarray((y_mask*x_img).astype(np.uint8))
            img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_x_seg.png"))
        elif last_activation == "sigmoid":
            for j in range(label.n_labels):
                label_name =label.name[j]

                img = Image.fromarray(seg_img[j])
                img.save(os.path.join(out_dir_valid, str(i).zfill(6) + label_name + "_seg.png"))

                img = Image.fromarray(x_img)
                img.save(os.path.join(out_dir_valid, str(i).zfill(6) + label_name + "_x.png"))

                y_mask = seg_img[j].copy()/255
                black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
                white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
                y_mask[black_pix,:] = [0.5,0.5,0.5]
                y_mask[white_pix,:] = [0.5,0.5,0.5]
                img = Image.fromarray((y_mask*x_img).astype(np.uint8))
                img.save(os.path.join(out_dir_valid, str(i).zfill(6) + label_name + "_x_seg.png"))
