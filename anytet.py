from anytree import Node, RenderTree

r = Node("actor(A)", vars="")
n1 = Node("role(A,M)", parent=r, vars="M")
n1 = Node("win(M,P)", parent=n1, vars="P")
n1 = Node("win(A,P1)", parent=r, vars="P1")

for pre, _, node in RenderTree(r):
    print("{}{},{}".format(pre, node.name, node.vars))


print(r.children)
