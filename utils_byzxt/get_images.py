from path import Path
def get_data(img_dir):
    img_dir = Path(img_dir)
    file_type = ['*.jpg', '*.png', '*.jpeg', '*.tif']
    imgs = []
    for ft in file_type:
        imgs += sorted(img_dir.glob(ft))
    images = sorted(imgs)
    return images