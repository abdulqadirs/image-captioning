from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from pathlib import Path

def save_captions(images_dir, output_dir, image_id, caption):
    """
    Reads the image and writes captions on them.

    Args:
        images_dir (Path): Path of images directory.
        output_dir (Path): Path of output directory.
        image_id (Path): Id of image to be read.
        caption (Path): Caption of given image.
    
    Raises:
        PathError: If the given paths are invalid.
    """
    # TODO (aq): check if the given paths are valid and don't hardcode the paths.
    # TODO (aq): Display multiline text on images.
    caption = caption[1:]
    caption = " ".join(caption)
    img_path = Path(images_dir / image_id[0])
    img = Image.open(img_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)
  
    draw.rectangle([(0, 0),(width, 12)], outline=(255, 0, 0), fill=(255, 255, 255))
    draw.multiline_text((0, 0), caption, fill=(0, 0, 0), spacing=4)
    ouput_img_path = Path(output_dir / 'results' / image_id [0])
    img.save(ouput_img_path, "JPEG")