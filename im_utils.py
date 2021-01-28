from __future__ import division
import numpy as np
from PIL import Image
from io import BytesIO


def pil2bio(im, fmt='PNG'):
    """Convert a PIL Image to a StringIO object """
    if im.mode == 'CMYK': # this causes an error
        im = im.convert('RGB')
    bio = BytesIO()
    im.save(bio, format=fmt)
    bio.seek(0)
    return bio


def bio2pil(bioimage):
    """Reverse of pil2bio """
    bio = BytesIO()
    bio.write(bioimage)
    bio.seek(0)
    image = Image.open(bio)
    return image


def scale_pad_to_square(im, square_size, pad_colour = 255, pad_mode = 'constant'):
    """ 
    This function applies following pre-processing steps on the image
        1. Resize the image to make it's larger dimension to square_size (keep AR)
        2. Pad the lower dimension by zeros up to the square size
    Args:
        im : image (PIL image)
        pad_colour: dcitonary of top, bottom , left, right rgb colour or constant
        pad_mode: refer numpy pad modes + edge_mean

    Returns:
        PIL image with the size of [square_size, square_size, 3]
    """

    #TODO woring on common RGB mode as lazy to do the logics for arrays everytime
    if im.mode != 'RGB':
        im = im.convert('RGB')

    if type(pad_colour) == int and pad_mode == 'constant':
        c = pad_colour
        pad_colour = {'top' : [c]*3, 'left' : [c]*3, 'right' : [c]*3, 'bottom' : [c]*3}

    if pad_mode == 'edge_mean':
        pad_mode = 'constant'
        im_arr = np.asarray(im)
        pad_colour = {'top': np.mean(im_arr[0], axis=0), 'bottom': np.mean(im_arr[-1], axis=0), 'left': np.mean(im_arr[:,0], axis=0), 'right': np.mean(im_arr[:,-1], axis=0)}

    w = im.size[0]
    h = im.size[1]

    assert w != 0 and h != 0, "input image resulted in size: %dx%d" % (w, h)
    npad = []

    if (w > h):
        w1 = int(square_size)
        h1 = int(square_size/w * h)

        pad0 = (square_size - h1) // 2
        pad1 = (square_size - h1) - pad0
        npad = ((pad0, pad1), (0,0))
        if pad_mode =='constant':
            pad_clr = [pad_colour['top'], pad_colour['bottom']]
    elif (w < h) :
        h1 = int(square_size)
        w1 = int(square_size/h * w)
        pad0 = (square_size - w1) // 2
        pad1 = (square_size - w1) - pad0
        npad = ((0, 0), (pad0, pad1))
        if pad_mode == 'constant':
            pad_clr = [pad_colour['left'], pad_colour['right']]
    else:
        padded_im = im    
        w1 = square_size
        h1 = square_size

    assert w1 != 0 and h1 != 0, "scaled image resulted in size: %dx%d" % (w1, h1)
    im = im.resize((w1, h1), Image.ANTIALIAS)

    # if image need to be padded
    if len(npad) > 0:
        im = np.asarray(im)
        padded_im = np.zeros(shape = (square_size, square_size,3), dtype= np.uint8)
        for i in range(3):
            if pad_mode == 'constant':
                padded_im[:,:,i] = np.pad(im[:,:,i], pad_width=npad, mode='constant', constant_values= (pad_clr[0][i],pad_clr[1][i]))
            else:
                padded_im[:,:,i] = np.pad(im[:,:,i], pad_width=npad, mode=pad_mode)

        output_im = Image.fromarray(padded_im, 'RGB')
    else:
        output_im = im

    wh = output_im.size

    assert wh[0] == square_size and wh[1] == square_size, "scaled image resulted in size: %dx%d" % (wh[0], wh[1])

    return output_im


def undo_scale_pad_to_square(im, orig_size, bboxes=[]):
    """Reverses the scale_pad_to_square function, i.e. crop then resize.
    Returns corrected image and bboxes

    im -- PIL image
    orig_size -- (width, height)
    bboxes -- list of bboxes from dabbox.py to correct
    """
    square_size = im.size[0]
    w = orig_size[0]
    h = orig_size[1]

    left = 0
    upper = 0
    right = im.size[0]
    lower = im.size[1]

    if (w > h):
        w1 = int(square_size)
        h1 = int(square_size/w * h)
        upper = (square_size - h1) // 2
        lower -= (square_size - h1) - upper
    else:
        h1 = int(square_size)
        w1 = int(square_size/h * w)
        left = (square_size - w1) // 2
        right -= (square_size - w1) - left
    im = im.crop((left, upper, right, lower))
    im = im.resize(orig_size, Image.ANTIALIAS)

    bboxes_fixed = []
    for bbox in bboxes:
        bboxes_fixed.append([
                (int(round(bbox[0] * square_size)) - left) / float(w1),
                (int(round(bbox[1] * square_size)) - upper) / float(h1),
                (int(round(bbox[2] * square_size)) - left) / float(w1),
                (int(round(bbox[3] * square_size)) - upper) / float(h1)
        ])

    return im, bboxes_fixed