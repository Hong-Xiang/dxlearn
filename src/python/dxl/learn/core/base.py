
class Host:
    """
    Object saving host information.
    """
    def __init__(self, cluster:str, index:int=0):
        """
        Parameters:

        - `cluster`: A string of host type, like 'master', 'worker', 'ps', etc.
        - `index`: Index of correspoding node given specific cluster.
        """
        self.cluster = cluster
        self.index = index
    
    def __eq__(self, h: 'Host'):
        return self.cluster == h.cluster and self.index == h.index