from doufo import singledispatch

@singledispatch()
def distribute(obj, cluster, host):
    pass