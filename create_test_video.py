import numpy as np
from skimage import data
import imageio.v3 as iio

chelsea = data.chelsea()
astronaut = data.astronaut()

frames = [chelsea for _ in range(10)]
iio.imwrite('/opt/molink/app/test_cat_video.mp4', np.stack(frames), fps=5)
print('Created test_cat_video.mp4:', frames[0].shape)

frames2 = [astronaut for _ in range(10)]
iio.imwrite('/opt/molink/app/test_astronaut_video.mp4', np.stack(frames2), fps=5)
print('Created test_astronaut_video.mp4:', frames2[0].shape)