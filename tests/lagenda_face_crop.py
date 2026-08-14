from PIL import Image

def crop_face(image_path: str, x0: int, y0: int, x1: int, y1: int,) -> Image.Image:
   
    img = Image.open(image_path)
    img = img.convert("RGB")  # normalize mode
 
    width, height = img.size
 
    
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Degenerate box: (x0={x0}, y0={y0}) -> (x1={x1}, y1={y1}). "
            f"x1 must be > x0 and y1 must be > y0. "
            f"If this trips constantly, your coordinate order/convention assumption is wrong."
        )
 
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"Box ({x0},{y0},{x1},{y1}) falls outside image bounds ({width}x{height}). "
            f"Check whether coordinates are normalized (0-1) instead of absolute pixels."
        )
 
    face = img.crop((x0, y0, x1, y1))
 
    # if output_path:
    #     face.save(output_path)
 
    return face