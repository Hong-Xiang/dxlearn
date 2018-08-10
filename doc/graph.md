Graph is a collections of Tensors, graphs, configs and info.

Features for config:

- provide access to default / external config by `info.name`;
- provide self.config(key) search-like config getter, from config of father graph.
- provide _default_config() classmethod;


Features for tenosrs:
- provide default collections of holding tensors: self.tensors, main interface of graph.

`self.tensors` is a dict of Tensors, which is provided as an interface to Graph.
Which means one may use ::

    g = Graph(...)
    g.run(key) 
    sess.run(g.tensor(key))
    
to run corresponding tensors.

Another useage is to substitute part of graph with other tensors. One may use ::

    g = SomeGraph(tensor_dict, ...)
    g.tensors = {'x': Tensor}
    g.run(key, feeds={'x': tensor, 'y': tensor2})
    g.x == g.tensor('x')
    g.run(key, feeds={g.x: tensor, g.y: tensor})

which is equivilant to ::

    sess.run(g.tensor(key).data, feeds={g.tensor(k).data:tensor for k in ['x', 'y']})


KEYS:

- `TENSOR`:
    - `MAIN`: Main purpose/effect of graph, thus the one which is fetched by
    by default, thus `g.run()` is eqvilant to `g.run(g.KEYS.TENSOR.MAIN)`.

    - tensor(key): -> Tensor


Provide The following methods:

- `g.tensor(key)`
- `g.graph(key)`
- `g.config(key)`

- `g.run(key)`


# graph maker design

Since our library targeting easily reuse and substitution of sub-graph,
there would be four common cases when constructing Graph with sub-graphs.

1. father graph is not going to be reused (e.g. for Graphs), subgraph is fixed
2. father graph is going to be reused (e.g. for Model), subgraph is fixed
3. father graph is not going to be reused, subgraph is configurable
4. father graph is going to be reused, subgraph is configurable

For case 1:
just directly code it in kernel:

``` python
def kernel(self):
    x = self.tensor('input')
    subg = SomeGraph(self.info.child_scope('sub'), tensors={'x': x})
    subg.make()
    y = subg.tensor('y')
```

For case 2:
Use `graphs` collection.



``` python
# subg is model
def kernel(self):
    x = self.tensor('input')
    subg = self.graph('sub', Conv2D(filters=32))
    y = subg(x)
```

For case 3:

``` python
def kernel(self):
    x = self.tensor('input')
    subg = self.graph('sub')
    subg.tensors['x'] = x
    subg.make()
    y = subg.tensor('y')
```

For case 4:

``` python
def kernel(self):
    x = self.tensor('input')
    subg = self.graph('sub')
    y = subg(x)
```


`GraphInfo`
`name`: extract config

`variable_scope`: VariableScope(name, reuse)

#`variable_scope`: variable_scope
#`reuse`: reuse

`DistributeGraphInfo(GraphInfo)`
`host`: Host(job_name, task_index, ip, port)
