# Objective

## Decouple config, architecure and implementation details

Configurations can be import through a external config tree, via `graph.config(key)` interface.

`tree` is for inherent, thus sub-graph can use configs from their father graphs' configuration as default.

.json -> import hyper-parameters/configurations (simple values (int, float, dict, list))
Useful for grid search of hyper-parameters.

.py -> import network architecture
Portable definition of network architecure.

## Hierarchy and modular design

A "multi-scale" network/model description.

Low level details about sub-blocks/models are hidden to higher-level graphs/models,
except their interface `graph.tensor(key)`, `graph.graph(key)`.

Low level model/graph is easy to substitute.
Fast development of new network architecture.

## Unified local/distribute interface

With minor modification, we can transfer a local graph to a distribute graph.

## Use OOP (python object) to represent variable scope, device scope

`Model` is safe to reuse transferring python object, without worrying about variable scope.
`DistributeGraphInfo` take care of device scope.
