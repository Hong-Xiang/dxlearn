`Model` is reuse-able `Graph`    
A special case of Graph, which all inputs are listed in inputs, i.e. no Tensor
created in constructing model will introduce external information, works like a
function. Note `Model` is not "pure" function since there maybe variables
for model itself.  

Model provide `__call__` method, which make reuse of Model much more easier.

the first time `__call__` was called, it will call `make()`.