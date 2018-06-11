based on `dxl.core.config`, its a implementation of config tree.

`CNode` is node to construct tree, which is a dict-like object, with an additional `child` member.
`CView(CNode)` is used for fuse query of `key` to get configuration.

`ConfigurableWithName` has a method `self.config(key)` is implemented by:

``` python
t = ... # construct config tree
o = ConfigurableWithName('x/y')
o.view == CView('x/y')

o.config(key) == o.view.search(key)
```