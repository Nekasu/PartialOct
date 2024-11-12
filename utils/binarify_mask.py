from PIL import Image

def binarify_mask(in_image_path_name, out_image_path_name):
    image = Image.open(in_image_path_name)
    
    pixels = image.load()
    
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            r, g, b = pixels[x, y]
            if (r <=10) and (g<=10) and (b<=10):
                pixels[x,y] = (0, 0, 0)
    
    image.save(out_image_path_name)