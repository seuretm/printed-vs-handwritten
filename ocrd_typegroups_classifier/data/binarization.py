from PIL import Image
from skimage import filters
from random import randint
import numpy as np

class Sauvola(object):
    """ Sauvola binarization data augmentation method compatible with PyTorch.
    """
    
    def __init__(self, min_r=2, max_r=10):
        """ 
            Parameters
            ----------
            
            min_r: int
                minimum radius 
            max_q: int
                maximum compression quality (ceiled to 100) - note that
                this value must not be lower than min_q.
        """
        
        self.min_r = max(min_r, 1)
        self.max_r = min(max_r, 50)
        assert self.min_r <= self.max_r

    def __call__(self, img):
        """ Returns a binarized version of the image
            
            Parameters
            ----------
            
            img: PIL.Image
            
            Returns
            -------
                res: PIL.Image
        """
        
        r = 1+2*randint(self.min_r, self.max_r)
        arr = np.array(img)
        t = filters.threshold_sauvola(arr, window_size=r)
        arr[arr<t]  =   0
        arr[arr>=t] = 255
        return Image.fromarray(arr)
    
    def __repr__(self):
        """ Returns a string description of the class """
        return self.__class__.__name__ + '(min_r={}, max_r={})'.format(self.min_r, self.max_r)


class Otsu(object):
    """ Sauvola binarization data augmentation method compatible with PyTorch.
    """
    
    def __init__(self):
        """ 
            Parameters
            ----------
            None
        """

    def __call__(self, img):
        """ Returns a binarized version of the image
            
            Parameters
            ----------
            
            img: PIL.Image
            
            Returns
            -------
                res: PIL.Image
        """
        
        arr = np.array(img.convert('LA'))
        try:
                t = filters.threshold_otsu(arr)
        except:
                return img # probably single-value image
        arr[arr<t]  =   0
        arr[arr>=t] = 255
        return Image.fromarray(arr).convert('RGB')
    
    def __repr__(self):
        """ Returns a string description of the class """
        return self.__class__.__name__ + '()'
