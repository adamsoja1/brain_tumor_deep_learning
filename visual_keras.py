import visualkeras
from KerasModels.unet_model import multi_unet_model



model = multi_unet_model(4, 160, 160, 4)

from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 40)
visualkeras.layered_view(model,draw_volume=False, legend=True, font=font) # selected font





model.summary()

from keras.utils import plot_model

plot_model(model)
