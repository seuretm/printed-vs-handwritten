from io import BytesIO
from PIL import Image
from random import randint

class QLoss(object):
    """ Quality loss-based data augmentation method compatible with PyTorch.
    
        Applied a random JPEG compression to the input PIL image. By default,
        the quality is between 1 and 100.
    """
    
    def __init__(self, min_q=1, max_q=100):
        """ 
            Parameters
            ----------
            
            min_q: int
                minimum compression quality (automatically floored to 1)
            max_q: int
                maximum compression quality (ceiled to 100) - note that
                this value must not be lower than min_q.
        """
        
        self.min_q = max(min_q, 1)
        self.max_q = min(max_q, 100)
        assert self.min_q <= self.max_q

    def __call__(self, img):
        """ Returns a copy of an image with JPG degradations
            
            Parameters
            ----------
            
            img: PIL.Image
            
            Returns
            -------
                res: PIL.Image
        """
        
        temp = BytesIO()
        q = randint(self.min_q, self.max_q)
        img.save(temp, format='jpeg', quality=q)
        res = Image.open(temp)
        return res
    
    def __repr__(self):
        """ Returns a string description of the class """
        return self.__class__.__name__ + '(min_q={}, max_q={})'.format(self.min_q, self.max_q)
