import matplotlib.pyplot as plt
import numpy as np
import json

# load  json from 'video/2024-01-13 08.38.31/meta.json'
# get movement_detection.raw_white_pixel_values
with open("video/2024-01-13 08.38.31/meta.json") as f:
    data = json.load(f)

# get the raw white pixel values
raw_white_pixel_values = data["movement_detection"]["raw_white_pixel_values"]
print(raw_white_pixel_values)


# plot the raw white pixel values
x = np.arange(0, len(raw_white_pixel_values))
y = raw_white_pixel_values

plt.plot(x, y)
plt.show()
