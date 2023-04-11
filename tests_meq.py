import string

from .meq import *


def build_test_graph():
    # Build a random-ish graph to test the algorithm against.
    # We should probably use the networkx library for building the graph (setting the "1", "2", "3" labels as node
    # attributes), then visualise it separately with graphviz; but, this seems to be overkill.
    test_graph = ExploredGraph(nodes=[])
    # We are abusing the "string" library to get a list of nodes with one value per alphabet letter.
    # If we ever need more nodes, this should be changed. The rest of the code should work with no changes though.
    test_node_names = [node for node in string.ascii_uppercase]
    for test_node_name in test_node_names:
        test_graph.add_node(ExploredGraphNode(name=test_node_name, edges=[]))
    node_count = test_graph.node_count()

    # First, add the transition (which has no label) from the end node ("final state") back to the start.
    test_graph.nodes[node_count - 1].add_edge(
        ExploredGraphNode.ExploredGraphEdge(to_node=test_graph.nodes[0], label=None)
    )

    # Remove the last node from the list, it only has the one transition that we added above.
    regular_nodes = test_graph.nodes[:-1]
    # Shuffle them to add some visible irregularity (in other words: switching the labels around makes no real
    # difference, but the result looks somewhat more random this way).
    random.shuffle(regular_nodes)

    # Connect at least one node to the final state, to make sure the final state is reachable in at least one way.
    node_connected_to_last = random.choice(regular_nodes)
    node_connected_to_last.add_edge(
        ExploredGraphNode.ExploredGraphEdge(
            to_node=test_graph.nodes[node_count - 1], label=3
        )
    )
    for i, regular_node in enumerate(regular_nodes):
        # This is "node_count - 2" because we removed the last element of the list.
        next_node = i + 1 if i < node_count - 2 else 0
        # Add a connection to another node, and at the end of the list, circle back to the start.
        # This will ensure a fully connected graph where each node is reachable
        # (except the last, which is not included in this loop, but we took care of it separately above).
        regular_node.add_edge(
            ExploredGraphNode.ExploredGraphEdge(
                to_node=regular_nodes[next_node], label=1
            )
        )
        # Then, connect to two other nodes. Use the full list and not regular_nodes, so as to have
        # more than one potential path to the final state.
        other_nodes = test_graph.nodes.copy()
        # But, do not create a self-loop.
        other_nodes.remove(regular_node)
        other_nodes_to_connect = random.choices(other_nodes, k=2)
        regular_node.add_edge(
            ExploredGraphNode.ExploredGraphEdge(
                to_node=other_nodes_to_connect[0], label=2
            )
        )
        # node_connected_to_last already has a "label 3" connection to the last node, so do not duplicate it.
        if regular_node != node_connected_to_last:
            regular_node.add_edge(
                ExploredGraphNode.ExploredGraphEdge(
                    to_node=other_nodes_to_connect[0], label=3
                )
            )

    # At this point, we should have a graph where:
    #  - each node is reachable from the start (including the final state)
    #  - there are three differently labeled edges from every node except the last
    #  - the last node is connected to the first by an un-labeled edge
    # Uncomment the line below to see it.
    # test_graph.visualize()
    return test_graph


class LocalAccessFunction(BaseAccessFunction):
    """
    A fake which allows us to test the exploration algorithm locally, without abusing the test server.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.graph = kwargs["graph"]
        self.first_node = self.graph.nodes[0]
        self.last_node = self.graph.nodes[self.graph.node_count() - 1]
        super().__init__(*args, **kwargs)

    def connect(self) -> str:
        # The first response from the server gives us the first node, without a specific request asking for it.
        self.graph.current_state = self.graph.nodes[0]
        return self.graph.current_state.name

    def disconnect(self) -> None:
        pass

    def query_current_node(self, label: int) -> str:
        next_node = self.graph.current_state.get_edge_by_label(label)
        if next_node == self.last_node:
            # Flip back to the start, return "A\nZ".
            self.graph.current_state = self.first_node
            return f"{self.last_node.name}\n{self.first_node.name}\n"

        self.graph.current_state = next_node
        return f"{next_node.name}\n"


def test_graph_discovery():
    test_graph = build_test_graph()
    access_function = LocalAccessFunction(graph=test_graph)
    discovered_graph = explore_graph(access_function=access_function)
    # Uncomment the line below to see the discovered graph.
    # discovered_graph.visualize()
    assert set(test_graph.node_names()) == set(discovered_graph.node_names())
    for test_node in test_graph.nodes:
        discovered_node = discovered_graph.get_node_by_name(test_node.name)
        for test_edge in test_node.edges:
            # There is some issue with the representation of the result, so this needs to be cast to str to compare?
            assert str(test_edge.to_node.name) == str(
                discovered_node.get_edge_by_label(test_edge.label)
            )
