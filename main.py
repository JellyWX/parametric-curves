import numpy as np
import cv2
from random import random
from math import cos, sin, radians
from time import time as utime

size = 60
max_degree = 8
border = 6
frames = 720
fps = 20
brush_size = 1

circle_scale = size / 2 - border

img = np.zeros([size * max_degree, size * max_degree, 3])

#paint = np.array([1, 0, 1])
#brush = np.array([1, 0, 0])

paint = (255, 255, 255)
brush = (0, 0, 255)
black = (0, 0, 0)


def brute():
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (size * max_degree, size * max_degree))

    def generate_trig_vals(a, b, theta):
        c = int( (cos(a * radians(theta)) * circle_scale) + circle_scale + border )
        s = int( (sin(b * radians(theta)) * circle_scale) + circle_scale + border )

        return c, s

    def generate_trig_sequence(a, b):
        vals = []

        for i in range(frames):
            vals.append(generate_trig_vals(a, b, i/(frames/360)))

        return vals


    all_sequences = []

    for degree_a in range(max_degree):
        new_row = []
        for degree_b in range(max_degree):
            seq = generate_trig_sequence(degree_a, degree_b)

            new_row.append(seq)

        all_sequences.append(new_row)


    for frame in range(1, frames):
        for degree_a, row in enumerate(all_sequences):
            for degree_b, sequence in enumerate(row):
                pos_x_old = size * degree_a + sequence[frame-1][0]
                pos_y_old = size * degree_b + sequence[frame-1][1]

                pos_x = size * degree_a + sequence[frame][0]
                pos_y = size * degree_b + sequence[frame][1]

                cv2.circle(img, (pos_x_old, pos_y_old), brush_size, paint, thickness=-1)
                cv2.circle(img, (pos_x, pos_y), brush_size, brush, thickness=-1)

        imgnorm = cv2.normalize(img, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        out.write(imgnorm)

brute()