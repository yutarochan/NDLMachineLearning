from pybrain.tools.tools.shortcuts import buildNetwork

net = buildNetwork([2, 3, 1])
net.activate([2, 1])
