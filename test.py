#!/usr/bin/python

from io import BytesIO
import re
import dlib
import cv2
import requests
import numpy as np
from skimage import io
from faceitize import replace_faces, get_subdiv
from faceitize.debug import draw_delauney, draw_rect

FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_PREDICTOR = dlib.shape_predictor(
    './shape_predictor_68_face_landmarks.dat')

def make_face_rect(box):
    return [box.left(), box.top(), box.right(), box.bottom()]


def landmarks_to_arr(landmarks):
    return np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 67)])


def landmarks_to_tpl(landmarks):
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 67)]

def extract_name(url):
    """Takes the a url and exracts the images name"""
    "\/.*$"
    # [^/]*\.*
    return re.search(r"[^\/]*\.(png|jpg|jpeg)", url)

urls = [
    'https://cdn2.newsok.biz/cache/r960-3202a662d2e709a776a3c3de2ef35555.jpg',
    'https://static01.nyt.com/images/2017/12/12/us/12Alabama1/12Alabama1-facebookJumbo.jpg',
    'https://cdn.cnn.com/cnnnext/dam/assets/171215023038-petersen-nominee-02-super-tease.jpg',
    'https://cdn.cnn.com/cnnnext/dam/assets/171207155934-mueller-fox-super-tease.jpg',
    'https://cdn.cnn.com/cnnnext/dam/assets/170516163634-donald-trump-051617-super-tease.jpg',
]

static = requests.get(
    'https://nyppagesix.files.wordpress.com/2017/07/170714_yang_nyp___manh_fed___dmx_3.jpg'
).content

static = io.imread(BytesIO(static))
static = cv2.cvtColor(static, cv2.COLOR_BGR2RGB)
static_copy = np.copy(static)

faces = FACE_DETECTOR(static, 1)

for _, face in enumerate(faces):
    # src_rect = make_face_rect(face)
    src_dlib_landmarks = LANDMARK_PREDICTOR(static, face)
    src_arr_landmarks = landmarks_to_arr(src_dlib_landmarks)
    src_tpl_landmarks = landmarks_to_tpl(src_dlib_landmarks)
    src_hull = [
        cow[0] for cow in cv2.convexHull(np.float32([src_tpl_landmarks]))
    ]
    src_subdiv = get_subdiv(src_tpl_landmarks)

    # Debug
    draw_delauney(static_copy, src_subdiv)
    # draw_rect(static_copy, src_rect)
    # cv2.imwrite('debug.jpg', static_copy)

for url in urls:
    # Get and load image into memory
    f = requests.get(url).content
    img = io.imread(BytesIO(f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = replace_faces(img, static, src_tpl_landmarks, landmark_predictor=LANDMARK_PREDICTOR)
    cv2.imwrite(extract_name(url)[0], img)
