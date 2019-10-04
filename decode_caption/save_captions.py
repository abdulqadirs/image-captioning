from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from pathlib import Path


def save_captions(images_dir, output_dir, image_id, caption):
    """
    Reads the image and writes captions on them

    Params
    ------
    - images_dir: path of images directory
    - output_dir: path of output directory
    - image_id: id of image to be read
    - caption: caption of given image

    Return
    ------    

    """
    caption = " ".join(caption)
    #print(caption)
    img_path = Path(images_dir / image_id[0])
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    #font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.multiline_text((0, 0), caption,fill=(255,255,255), spacing=4)
    ouput_img_path = Path(output_dir / 'results' / image_id [0])
    img.save(ouput_img_path, "JPEG")

