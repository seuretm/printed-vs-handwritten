import _io
import torch
import pickle
from torchvision import transforms
from PIL import Image

from ocrd_typegroups_classifier.data.classmap import ClassMap
from ocrd_typegroups_classifier.data.classmap import IndexRemap



class TypegroupsClassifier:
    """ Class wrapping type group information and a classifier.
    
        Attributes
        ----------
        
        classMap: ClassMap
            Maps class names to indices corresponding to what the network
            outputs.
        network: PyTorch network
            Classifier
        dev: str
            Device on which the data must be processed
    
    """
    
    def __init__(self, groups, network, device=None):
        """ Constructor of the class.
        
            Parameters
            ----------
            
            groups: map string to int
                Maps names to IDs with regard to the network outputs;
                note that several names can point to the same ID, but
                the inverse is not possible.
            network:PyTorch network
                This network has to have the same interface as the
                VRAEC, return three values when being called: the
                classification result, a variational loss, and a feature
                vector.
            device: str
                Device on which the data has to be processed; if not set,
                then either the cpu or cuda:0 will be used.
        
        """
        
        self.classMap = ClassMap(groups)
        self.network = network
        if device is None:
            self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = device
        network.to(self.dev)
    
    @classmethod
    def load(cls, input):
        """ Loads a type groups classifier from a file
            
            Parameters
            ----------
            input: string or file
                File or path to the file from which the instance has to
                be loaded.
        
        """
        
        if type(input) is str:
            f = open(input, 'rb')
            res = cls.load(f)
            f.close()
            return res
        if not type(input) is _io.BufferedReader:
            raise Exception('TypegroupsClassifier.load() requires a string or a file')
        res = pickle.load(input)
        # If trained with CUDA and loaded on a device without CUDA
        res.dev = torch.device(res.dev if torch.cuda.is_available() else "cpu")
        res.network.to(res.dev)
        return res
        
    def save(self, output):
        """ Stores the instance to a file
        
            Parameters
            ----------
                output: string or file
                    File or path to the file to which the instane has to
                    be stored.
        """
        
        if type(output) is str:
            f = open(output, 'wb')
            self.save(f)
            f.close()
            return
        if not type(output) is _io.BufferedWriter:
            raise Exception('save() requires a string or a file')
        # Moving the network to the cpu so that it can be reloaded on
        # machines which do not have CUDA available.
        self.network.to("cpu")
        pickle.dump(self, output)
        self.network.to(self.dev)
    
    def filter(self, sample, label):
        """ Removes data with unknown type groups
            
            Parameters
            ----------
                sample: PyTorch tensor
                    Tensor of inputs for the network
                label: PyTorch tensor
                    Tensor of class IDs, the unknown ones being set to -1
            
            Returns
            -------
                sample, label
                    The input tensors without the ones having a -1 label
        """
        
        selection = label!=-1
        return sample[selection], label[selection]
    
    def run(self, pil_image, stride, batch_size=32, score_as_key=False):
        return self.classify(pil_image, stride, batch_size, score_as_key)
    
    def classify(self, pil_image, stride, batch_size, score_as_key=False, max_width=1000):
        """ Classifies a PIL image, returning a map with class names and
            corresponding scores.
            
            Parameters
            ----------
                pil_image: PIL image
                    Image to classify
                stride: int
                    The CNN is applied patch-wise; this parameter
                    corresponds to the offset between two patches
                batch_size: int
                    Number of patches which can be processed at the same
                    time by the hardware. If no GPU is used, then a
                    value of 1 is fine.
                score_as_key: bool
                    Use scores, instead of class names, as key for the
                    result map.
            
            Returns
            -------
                A map between class names and scores, or scores and
                class names, depending on whether score_as_key is true
                or false.
        """
        
        if pil_image.size[0]>max_width:
            pil_image = pil_image.resize((max_width, round(pil_image.size[1]*float(max_width)/pil_image.size[0])), Image.BILINEAR)
        crop_size = min(224, pil_image.size[0])
        crop_size = min(crop_size, pil_image.size[1])
        tensorize = transforms.ToTensor()
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            score = 0
            processed_samples = 0
            batch = []
            for x in range(0, pil_image.size[0], stride):
                for y in range(0, pil_image.size[1], stride):
                    crop = tensorize(pil_image.crop((x, y, x+crop_size, y+crop_size)))
                    batch.append(crop)
                    if len(batch) >= batch_size:
                        tensors = torch.stack(batch).to(self.dev)
                        out = self.network(tensors)
                        score += out.sum(0)
                        processed_samples += len(batch)
                        batch = []
            if batch:
                tensors = torch.stack(batch).to(self.dev)
                out = self.network(tensors)
                score += out.sum(0)
                processed_samples += len(batch)
                batch = []
        if was_training:
            self.network.train()
        score /= processed_samples
        res = {}
        for cl in self.classMap.cl2id:
            cid = self.classMap.cl2id[cl]
            if cid == -1:
                continue
            res[cl] = score[cid].item()
        if score_as_key:
            res = {s: c for c, s in res.items()}
        return res
    
    def __repr__(self):
        """ returns a string description of the instance """
        
        format_string = self.__class__.__name__ + '('
        format_string += '\n ClassMap: %s' % self.classMap
        format_string += '\n Network:'
        if self.network is None:
            format_string += '\n  None'
        else:
            format_string += '\n%s\nEnd of network\n' % self.network
        return format_string+'\n)'
    
