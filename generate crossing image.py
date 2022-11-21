import cv2
import pandas as pd
from PIL import Image
from Environments import make_intersection

if __name__ == '__main__':
    env = make_intersection()
    env.reset()
    img = env.render()
    cv2.imwrite('intersection.jpeg', img)


