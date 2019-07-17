class IndexRemap:
    """ Utility class for remapping class indices.
    
        Attributes
        ----------
        id2id: dictionary
            map from source to target index
    """
    
    def __init__(self, id2id):
        """ 
            Parameters
            ----------
            
            id2id: dictionary int to int
                map from source to target index
        """
        self.id2id = id2id
    
    def __call__(self, n):
        """ Remaps an index, returns -1 if the input index is not known
        
        """
        if not n in self.id2id:
            return -1
        return self.id2id[n]
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for k in self.id2id:
            format_string += '\n %d:%d' % (k, self.id2id[k])
        return format_string+'\n)'


class ClassMap:
    """ Class wrapping type group information and a classifier.
    
        Attributes
        ----------
        
        cl2id: dictionary string to int
            Maps a class name to a class number
        id2cl: dictionary int to string
            Maps a class number to a class name
    """
    
    def __init__(self, basemap):
        """ Constructor of the class.
        
            Parameters
            ----------
            
            basemap: map string to int
                Maps names to IDs with regard to the network outputs;
                note that several names can point to the same ID, but
                the inverse is not possible.
        """
        
        self.cl2id = {}
        self.id2cl = {}
        for c in basemap:
            self.cl2id[c] = basemap[c]
            self.id2cl[basemap[c]] = c
    
    def forget_class(self, target):
        del self.id2cl[self.cl2id[target]]
        del self.cl2id[target]
       
    def get_target_transform(self, dataset_classes):
        """ Creates a transform from a map (class name to id) to the
            set of IDs used by this class map. Unmatched classes are
            mapped to -1.
            
            This method is useful for producing target transforms for
            PyTorch ImageFolder. Proceed as follows:
                imf = ImageFolder('/path')
                imf.target_transform = cm.get_target_transform(imf.class_to_idx)
            where cm is a ClassMap instance.
        """
        
        idmap = {}
        for c in dataset_classes:
            if not c in self.cl2id:
                idmap[dataset_classes[c]] = -1
            else:
                idmap[dataset_classes[c]] = self.cl2id[c]
        return IndexRemap(idmap)
    
    def __repr__(self):
        return '%s' % self.cl2id
