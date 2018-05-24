class DatasetsManager:
    """
    A helper class to manage multiple datasets, in most cases, it is used to
    resolve dataset paths, construct dataset objects from name.
    """
    @classmethod
    def path_datasets(cls):
        """
        Returns path of all datasets (commonly used shared ones).
        """
        raise NotImplementedError
    
    @classmethod
    def get_dataset(cls, config):
        """
        Returns a Dataset object 
        """
        raise NotImplementedError

