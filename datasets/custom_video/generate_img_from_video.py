import numpy as np
import glob
import cv2
import h5py
from os import path
from PIL import Image
import matplotlib.pyplot as plt

video_dir_name = './datasets/custom_video/'
video_name_list = glob.glob(video_dir_name + "*.mp4")
print(video_name_list)

video_frame, size = [], 256
frame_org_width, frame_org_height = [], []
frame_start_idx, frame_end_idx = [f"{0:06}"], []
for video_idx, video_file in enumerate(video_name_list):
  vidcap = cv2.VideoCapture(video_file)

  count = int(frame_start_idx[video_idx])
  while (True):
    success, image = vidcap.read()
    if not success : break
    count += 1

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    org_w, org_h = np.shape(image)[0], np.shape(image)[1]
    image = cv2.resize(image, (size, size))

    #plt.imshow(image)
    #plt.show()
    video_frame.append(image)
    frame_org_width.append(org_w)
    frame_org_height.append(org_h)

  print(f"finished video {video_idx}")
  frame_end_idx.append(f"{count-1:06}")    
  if video_idx != len(video_name_list)-1 :
    frame_start_idx.append(f"{count:06}")

print(np.shape(video_frame))
print(frame_start_idx)
print(frame_end_idx)

for i, img in enumerate(video_frame):
  img = Image.fromarray(img.astype('uint8'), 'RGB')
  img.save(f"{video_dir_name}images/{i:06}.png",'png')

with h5py.File(path.join(video_dir_name, f'vddb_{size}.h5'), 'w') as h5_file:
  #h5_file.create_dataset('video_frame', data=video_frame)
  h5_file.create_dataset('video_start_idx', data=frame_start_idx)
  h5_file.create_dataset('video_end_idx', data=frame_end_idx)