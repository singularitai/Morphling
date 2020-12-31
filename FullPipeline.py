import gc
import subprocess
import multiprocessing


def synthesize(speaker_id, text, filename, model_used="libritts", sigma=0.8, n_frames=65536):
    import os
    from os.path import exists, join, basename, splitext
    from scipy.io.wavfile import write
    import json
    import torch
    import numpy as np
    import sys
    import matplotlib
    import matplotlib.pylab as plt

    from glow import WaveGlow
    from flowtron import Flowtron
    from data import Data

    plt.rcParams["axes.grid"] = False

    sys.path.insert(0, 'tacotron2')
    sys.path.insert(0, 'tacotron2/waveglow')

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # read config
    config = json.load(open('config.json'))
    data_config = config["data_config"]
    model_config = config["model_config"]
    # there are 123 speakers

    if model_used == "libritts":
        data_config[
            'training_files'] = 'filelists/{}_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt'.format(
            model_used)
        model_config['n_speakers'] = 123
    else:
        data_config['training_files'] = 'filelists/ljs_audiopaths_text_sid_train_filelist.txt'
        model_config['n_speakers'] = 1
        speaker_id = 0

    data_config['validation_files'] = data_config['training_files']

    # load waveglow
    waveglow = torch.load("models/waveglow_256channels_universal_v5.pt")['model'].cuda().eval()
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()
    _ = waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    state_dict = torch.load("models/flowtron_{}.pt".format(model_used), map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    _ = model.eval()

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

    speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
    text = trainset.get_text(text).cuda()
    speaker_vecs = speaker_vecs[None]
    text = text[None]

    print(speaker_vecs)
    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
        mels, attentions = model.infer(residual, speaker_vecs, text)

    audio = waveglow.infer(mels.half(), sigma=0.8).float()
    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()

    del model
    del waveglow

    torch.cuda.empty_cache()

    del torch
    gc.collect()

    write(filename, 22050, audio)


# In[2]:


def rect_to_bb(d):
    x = d.rect.left()
    y = d.rect.top()
    w = d.rect.right() - x
    h = d.rect.bottom() - y
    return (x, y, w, h)


def calcMaxArea(rects):
    max_cords = (-1, -1, -1, -1)
    max_area = 0
    max_rect = None
    for i in range(len(rects)):
        cur_rect = rects[i]
        (x, y, w, h) = rect_to_bb(cur_rect)
        if w * h > max_area:
            max_area = w * h
            max_cords = (x, y, w, h)
            max_rect = cur_rect
    return max_cords, max_rect


def face_detect(images, args):
    import scipy, cv2, os, sys, argparse, audio
    import dlib, json, h5py, subprocess
    from tqdm import tqdm
    detector = dlib.cnn_face_detection_model_v1(args.face_det_checkpoint)

    batch_size = args.face_det_batch_size

    predictions = []
    for i in tqdm(range(0, len(images), batch_size)):
        predictions.extend(detector(images[i:i + batch_size]))

    results = []
    pady1, pady2, padx1, padx2 = list(args.pads)[0]
    for rects, image in zip(predictions, images):
        (x, y, w, h), max_rect = calcMaxArea(rects)
        if x == -1:
            results.append([None, (-1, -1, -1, -1), False])
            continue
        y1 = max(0, y + pady1)
        y2 = min(image.shape[0], y + h + pady2)
        x1 = max(0, x + padx1)
        x2 = min(image.shape[1], x + w + padx2)
        face = image[y1:y2, x1:x2, ::-1]  # RGB ---> BGR

        results.append([face, (y1, y2, x1, x2), True])

    del detector  # make sure to clear GPU memory for LipGAN inference
    return results


def datagen(frames, mels, args):
    import numpy as np
    import scipy, cv2, os, sys, argparse, audio
    import dlib, json, h5py, subprocess
    from tqdm import tqdm

    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if not args.static:
        face_det_results = face_detect([f[..., ::-1] for f in frames], args)  # BGR2RGB for CNN face detection
    else:
        face_det_results = face_detect([frames[0][..., ::-1]], args)

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords, valid_frame = face_det_results[idx].copy()
        if not valid_frame:
            print("Face not detected, skipping frame {}".format(i))
            continue

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.lipgan_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def getfps(video_name):
    import cv2
    video = cv2.VideoCapture(video_name)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        video_fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        video_fps = video.get(cv2.CAP_PROP_FPS)

    video.release()

    return video_fps


def generatelipgan(audio_filename, video_name):
    from os import listdir, path
    import numpy as np
    import scipy, cv2, os, sys, argparse, audio
    import dlib, json, h5py, subprocess
    from tqdm import tqdm
    # import keras
    import tensorflow as tf
    import tensorflow.keras as k
    from tensorflow.python.framework import ops
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model

    from dotmap import DotMap

    import os
    from os.path import exists, join, basename, splitext
    from PIL import Image
    import sys
    import matplotlib.pyplot as plt

    try:
        video_fps = getfps(video_name)
        print(video_fps)
    except:
        video_fps = 30

    parser = dict()
    parser['description'] = 'Code to generate talking face using LipGAN'
    parser['checkpoint_path'] = "logs/lipgan_residual_mel.h5"
    parser['model'] = 'residual'
    parser['face_det_checkpoint'] = 'logs/mmod_human_face_detector.dat'
    parser['face'] = video_name
    parser['audio'] = audio_filename
    parser['results_dir'] = 'results/'
    parser['static'] = False
    parser['fps'] = video_fps
    parser['max_sec'] = 240.
    parser['pads'] = [0, 0, 0, 0],
    parser['face_det_batch_size'] = 1
    parser['lipgan_batch_size'] = 8
    parser['n_gpu'] = 1
    parser['img_size'] = 96
    args = DotMap(parser)

    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    fps = args.fps
    mel_step_size = 27
    mel_idx_multiplier = 80. / fps

    if args.model == 'residual':
        from generator import create_model_residual as create_model

    else:
        from generator import create_model as create_model

    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
    else:
        video_stream = cv2.VideoCapture(args.face)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
            if len(full_frames) % 2000 == 0: print(len(full_frames))

            if len(full_frames) * (1. / fps) >= args.max_sec: break

        print("Number of frames available for inference: " + str(len(full_frames)))

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan!')

    mel_chunks = []
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    batch_size = args.lipgan_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, args)

    video_name = audio_filename.replace(".wav", "")

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(np.ceil(
                                                                        float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = create_model(args, mel_step_size)
            print("Model Created")

            model.load_weights(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(path.join(args.results_dir, video_name + ".avi"),
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        pred = model.predict([img_batch, mel_batch])
        pred = pred * 255

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p, (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio,
                                                               path.join(args.results_dir, video_name + ".avi"),
                                                               path.join(args.results_dir, video_name + "_voice.avi"))
    subprocess.call(command, shell=True)

    command = 'ffmpeg -y -loglevel panic -i {} {}'.format(path.join(args.results_dir, video_name + "_voice.avi"),
                                                          video_name + ".mp4")
    subprocess.call(command, shell=True)

    del pred
    del model

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(3)

    try:
        tf.reset_default_graph()

    except:
        ops.reset_default_graph()

    del k
    del tf

    # from numba import cuda
    # cuda.select_device(0)
    # cuda.close()

    for clear in range(20):
        gc.collect()


# In[3]:
def firstOrder(audio_filename, image):
    import imageio
    import numpy as np
    from skimage.transform import resize
    import warnings
    import subprocess
    from demo import load_checkpoints
    from demo import make_animation
    from skimage import img_as_ubyte
    from ISR.models import RRDN
    import torch
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                              checkpoint_path='first-order-motion-model/vox-cpk.pth.tar')

    video_name = audio_filename.replace(".wav", "")
    warnings.filterwarnings("ignore")

    source_image = imageio.imread(image)

    driving_video = imageio.mimread(video_name + ".mp4", memtest="4096MB")

    video_fps = getfps(video_name + ".mp4")
    print(video_fps)

    # Resize image and video to 256x256

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    del generator
    del kp_detector
    torch.cuda.empty_cache()

    del torch
    gc.collect()

    # save resulting video
    rdn = RRDN(weights='gans')
    imageio.mimsave(video_name + "_generated.mp4",
                    [img_as_ubyte(rdn.predict(frame * 255.) / 255.) for frame in predictions], fps=video_fps)
    # video can be downloaded from /content folder

    command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_filename, video_name + "_generated.mp4",
                                                               video_name + "_voice.mp4")
    subprocess.call(command, shell=True)

    command = 'ffmpeg -y -loglevel panic -i {} {}'.format(video_name + "_voice.mp4", "final/" + video_name + ".mp4")
    subprocess.call(command, shell=True)

    del rdn
    gc.collect()


def delete_files(filename, mode=0):
    import os
    import time

    for ending in ["_voice.mp4", ".wav", "_generated.mp4", ".mp4"]:
        if mode > 1 and ending == ".wav":
            continue
        else:
            file = filename.replace(".wav", ending)
            os.remove(file)

    for ending in [".avi", "_voice.avi"]:
        file = "results/" + filename.replace(".wav", ending)
        os.remove(file)

    gc.collect()


def getAllFiles(path):
    import os
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    return files


def getImage():
    import io
    import requests
    from PIL import Image
    from matplotlib import pyplot as plt

    try:
        r = requests.get("https://thispersondoesnotexist.com/image", timeout=0.25)
        content = r.content
        image = Image.open(io.BytesIO(content))
        image.save("temp.png")
        image.save("first-order-motion-model/temp.png")
    except:
        print("image retrieval error")
        time.sleep(1)
        getImage()



def genderEstimate(face_detector, age_gender_detector, emotion_detector):
    import numpy as np
    from PIL import Image

    faces, boxes, scores, landmarks = face_detector.detect_align(
        np.array(Image.open("first-order-motion-model/temp.png")))
    genders, ages = age_gender_detector.detect(faces)

    list_of_emotions, probab = emotion_detector.detect_emotion(faces)
    smiling = False
    if "happy" in list_of_emotions:
        if probab[0] > 0.666:
            smiling = True

    return genders[0], ages[0], smiling


# In[4]:



if __name__ == '__main__':
    mode = 1

    full_text = ""
    #full_text += "What if a computer could animate a face?."
    #full_text += "using only text?."
    #full_text += "What if we could animate this face?."
    #full_text += "Or how about this face?."
    #full_text += "And, what about this one?."

    #full_text += "How about a famous face?."
    #full_text += "What if that face, was your face?."
    #full_text += "Or that of an evil robot."
    #full_text += "Or the face of a human that has never existed?."
    #full_text += "Would you be able to tell?."

    #full_text += "What if I told you that artificial intelligence created All of the human faces you have seen."
    #full_text += "All of the voices you have heard."
    #full_text += "All of lip movements on the faces."
    #full_text += "It translated all of the facial expressions."
    #full_text += "And composed the background music you can hear."

    #full_text += "The A I created nearly everything in this video!."

    #full_text += "What else can artificial intelligence do?."
    full_text += "this is a g u i test."
    #full_text += "Welcome to the future."
    #full_text += "Welcome to the Singularity!."

    aim_gender = ""
    audio_folder = "audios/"

    if mode == 0 or mode == 1:
        import random

        for k, text in enumerate(full_text.split(".")):

            if len(text) > 1:

                gender = ""
                age = 0
                smiling = True
                if mode == 1:

                    if k == 0:
                        aim_gender = "Male"
                    elif k % 3 == 0:
                        aim_gender = "Male"
                    else:
                        aim_gender = "Female"

                    print(aim_gender)
                    from Retinaface.Retinaface import FaceDetector
                    from AgeGender.Detector import AgeGender
                    from FacialExpression.FaceExpression import EmotionDetector
                    import time
                    face_detector = FaceDetector(name='mobilenet', weight_path='Retinaface/Models/mobilenet.pth',
                                                 device='cpu')
                    age_gender_detector = AgeGender(name='full', weight_path='AgeGender/Models/ShufflenetFull.pth',
                                                    device='cpu')
                    emotion_detector = EmotionDetector(name='densnet121',
                                                       weight_path='FacialExpression/models/densnet121.pth',
                                                       device='cpu')
                    while age < 18:
                        gender = ""
                        age = 0
                        smiling = True
                        while gender != aim_gender:
                            #while smiling != False:
                            getImage()
                            gender, age, smiling = genderEstimate(face_detector, age_gender_detector, emotion_detector)
                            time.sleep(5)

                file = text

                if (len(file) > 64):
                    file = file[:64]

                filename = file.replace(".", "").replace("?", "").replace(" ", "_") + ".wav"

                speakers = [1069, 1088, 1116, 118, 1246, 125, 1263, 1502, 1578, 1841, 1867, 196, 1963, 1970, 200, 2092,
                            2136, 2182, 2196, 2289, 2416, 2436, 250, 254, 2836, 2843, 2911, 2952, 3240, 3242, 3259,
                            3436, 3486, 3526, 3664, 374, 3857, 3879, 3982, 3983, 40, 4018, 405, 4051, 4088, 4160, 4195,
                            4267, 4297, 4362, 4397, 4406, 446, 460, 4640, 4680, 4788, 5022, 5104, 5322, 5339, 5393,
                            5652, 5678, 5703, 5750, 5808, 587, 6019, 6064, 6078, 6081, 6147, 6181, 6209, 6272, 6367,
                            6385, 6415, 6437, 6454, 6476, 6529, 669, 6818, 6836, 6848, 696, 7059, 7067, 7078, 7178,
                            7190, 7226, 7278, 730, 7302, 7367, 7402, 7447, 7505, 7511, 7794, 78, 7800, 8051, 8088, 8098,
                            8108, 8123, 8238, 83, 831, 8312, 8324, 8419, 8468, 8609, 8629, 87, 8770, 8838, 887]

                female_speakers = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 20, 22, 30, 31, 33, 35, 37, 38,
                                   39, 40, 43, 44, 45, 46, 49, 54, 55, 57, 60, 61, 62, 67, 69, 70, 72, 75, 77, 78, 81,
                                   83, 84, 87, 88, 90, 91, 96, 101, 102, 104, 105, 109, 110, 113, 116, 119, 122]
                male_speakers = [3, 11, 16, 21, 23, 25, 26, 27, 28, 29, 32, 34, 36, 41, 42, 47, 50, 53, 56, 58, 59, 63,
                                 65, 68, 71, 73, 74, 76, 79, 82, 85, 86, 89, 92, 93, 98, 99, 106, 107, 108, 115, 117,
                                 120, 121]
                bad_female_speakers = [17, 24, 48, 51, 95, 111, 114]
                bad_male_speakers = [19, 52, 64, 66, 80, 94, 97, 100, 103, 112, 118]

                voice_model = "ljs"
                if mode == 1:
                    voice_model = "libritts"

                    if gender == "Female":
                        index = int(random.uniform(0, len(female_speakers) - 0.5))
                        chosen_speaker = speakers[female_speakers[index]]
                    else:
                        index = int(random.uniform(0, len(male_speakers) - 0.5))
                        chosen_speaker = speakers[male_speakers[index]]

                print(gender, "|" ,age, "|", text)
                p = multiprocessing.Process(target=synthesize, args=(chosen_speaker, text, filename, voice_model,))
                p.start()
                p.join()
                gc.collect()

                videos = ["first-order-motion-model/leocut.mp4", "first-order-motion-model/leo.mp4",
                          "first-order-motion-model/00.mp4", "first-order-motion-model/04.mp4",
                          "first-order-motion-model/08.mp4", "first-order-motion-model/10-backward.mp4"]

                index = int(random.uniform(0, len(videos) - 0.5))
                chosen_video = videos[index]

                #generatelipgan(filename, chosen_video)

                p = multiprocessing.Process(target=generatelipgan, args=(filename, chosen_video,))
                p.start()
                p.join()
                gc.collect()

                replacement_image = "first-order-motion-model/fembotII.png"
                if mode == 1:
                    if "robot" in text:
                        if gender == "Male":
                            replacement_image = "first-order-motion-model/Robot.png"
                        else:
                            replacement_image = "first-order-motion-model/fembotII.png"
                    else:
                        replacement_image = "first-order-motion-model/temp.png"

                p = multiprocessing.Process(target=firstOrder, args=(filename, replacement_image,))
                p.start()
                p.join()
                gc.collect()

                try:
                    delete_files(filename, mode)
                    gc.collect()
                except:
                    pass


    elif mode == 2:

        files = getAllFiles(audio_folder)
        print(files)

        for f in files:
            filename = f
            p = multiprocessing.Process(target=generatelipgan, args=(filename, "leocut.mp4",))
            p.start()
            p.join()
            gc.collect()

            p = multiprocessing.Process(target=firstOrder, args=(filename, "first-order-motion-model/Robot.png",))
            p.start()
            p.join()
            gc.collect()

            delete_files(filename, mode)
            gc.collect()
