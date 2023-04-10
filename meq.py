"""
Run this from CLI as:
python -c 'import meq; meq.run_exploration()'

However, this cannot connect to the test server as it gets an SSL error:
  File "/usr/lib64/python3.10/ssl.py", line 1342, in do_handshake
    self._sslobj.do_handshake()

Trying http (no SSL) fails differently:
ib3.exceptions.MaxRetryError: HTTPConnectionPool(host='20.28.230.252', port=65432):
Max retries exceeded with url: /3 (Caused by ProtocolError('Connection aborted.', BadStatusLine('A\n')))

Trying directly from command line:
curl http://20.28.230.252:65432
curl: (1) Received HTTP/0.9 when not allowed

curl https://20.28.230.252:65432 --> just times out.

It seems that the server is GRPC but to connect to it, more information is needed.
"""

import dataclasses
import random
from abc import abstractmethod

import graphviz
import requests

EXPECTED_EDGE_VALUES = [1, 2, 3]

# We should be able to explore 76 edges with a maximum cost of 26 transitions each, equal to 1976.
# However, this supposes an optimized algorithm always finding the optimal path to the next unexplored node.
# At 20x, we can just brute-force it, which seems to work OK on this scale. In practice, it succeeds in much fewer steps
# because in its "random walk" the algorithm always picks the unexplored edges of a node, if they exist.
# Obviously, it will not scale to millions of nodes and connections, for which we are going to need shortcuts
# (aka "heuristics").
# See note for explored_graph.current_path below.
MAX_EXPECTED_STEPS = 39520

CONNECTION_IP = "20.28.230.252"
CONNECTION_PORT = 65432


@dataclasses.dataclass()
class ExploredGraphNode:
    @dataclasses.dataclass()
    class ExploredGraphEdge:
        to_node: "ExploredGraphNode"
        label: int | None

        def __repr__(self) -> str:
            return f"{self.label} -> {self.to_node}"

    name: str
    edges: list[
        ExploredGraphEdge
    ]  # Outgoing edges only; incoming edges will be recorded at the node they start from.

    def get_edge_labels(self):
        return [edge.label for edge in self.edges] if self.edges else []

    def add_edge(self, edge: ExploredGraphEdge):
        # Make sure we do not duplicate a label.
        # In a proper "industrial" environment, an attempt to duplicate the same label but pointing to a different
        # node should raise an exception.
        for existing_edge in self.edges:
            if existing_edge.label == edge.label:
                return existing_edge

        self.edges.append(edge)
        return edge

    def get_edge_by_label(self, label):
        # Given a label, return the name of the node to which the edge with this label points.
        for edge in self.edges:
            if edge.label == label:
                return edge.to_node

        return None

    def get_unexplored_edge_labels(self) -> list[int]:
        return list(set(EXPECTED_EDGE_VALUES) - set(self.get_edge_labels()))

    def is_explored(self) -> bool:
        return set(self.get_edge_labels()) == set(EXPECTED_EDGE_VALUES)

    def is_last(self) -> bool:
        if len(self.edges) == 1 and self.edges[0].label is None:
            return True

        return False

    def __repr__(self) -> str:
        return f"{self.name}"


@dataclasses.dataclass()
class ExploredGraphPath:
    from_node: ExploredGraphNode
    to_node: ExploredGraphNode
    route: list[ExploredGraphNode.ExploredGraphEdge]


@dataclasses.dataclass()
class ExploredGraph:
    nodes: list[ExploredGraphNode] | None
    current_state: ExploredGraphNode
    current_path: ExploredGraphPath
    known_paths: list[ExploredGraphPath]

    def __init__(self, nodes) -> None:
        self.nodes = nodes
        self.current_state = self.nodes[0] if self.nodes else None
        self.current_path = ExploredGraphPath(
            from_node=self.current_state, to_node=self.current_state, route=[]
        )
        self.known_paths = []

    def add_node(self, node: ExploredGraphNode):
        self.nodes.append(node)
        return node

    def node_count(self):
        return len(self.nodes)

    def node_names(self):
        return [node.name for node in self.nodes]

    def get_node_by_name(self, name):
        for node in self.nodes:
            if str(node.name) == str(name):
                return node

        return None

    def get_total_edge_count(self):
        total_edge_count = 0
        for node in self.nodes:
            total_edge_count += len(node.edges)

        return total_edge_count

    def add_known_path(self, known_path: ExploredGraphPath):
        self.known_paths.append(known_path)

    def is_fully_explored(self):
        for node in self.nodes:
            # Check all nodes except the last, which only has one edge.
            if not node.is_explored() and not node.is_last():
                return False

        return True

    def visualize(self):
        display_graph = graphviz.Digraph()
        for node in self.nodes:
            for edge in node.edges:
                display_graph.edge(
                    str(node),
                    str(edge.to_node),
                    label=str(edge.label) if edge.label else "",
                )

        display_graph.view()

    def __eq__(self, other) -> bool:
        if self.node_count() == other.node_count():
            return False

        return True


class BaseAccessFunction:
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def connect(self) -> str:
        pass

    @abstractmethod
    def query_current_node(self, label: int) -> str:
        pass


class AccessFunction(BaseAccessFunction):
    def __init__(self, *args, **kwargs) -> None:
        # Start a session with the test server.
        self.session = requests.session()
        super().__init__(*args, **kwargs)

    def connect(self) -> str:
        # The first response from the server gives us the first node, without a specific request asking for it.
        server_response = self.session.get(
            f"https://{CONNECTION_IP}:{CONNECTION_PORT}", timeout=2, verify=False
        )
        return server_response.content

    def query_current_node(self, label: int) -> str:
        # We cannot select which node to query - it is always the "current state" of the state machine.
        # We can only send it an edge label, which will get the state machine to transition to a new state.
        server_response = self.session.get(
            f"https://{CONNECTION_IP}:{CONNECTION_PORT}/{label}"
        )
        return server_response.content


def explore_graph(access_function: BaseAccessFunction) -> ExploredGraph:
    explored_graph = ExploredGraph(nodes=[])
    number_passes = 0
    number_steps = 0

    def get_next_edge_label() -> int:
        edges_to_explore = explored_graph.current_state.get_unexplored_edge_labels()
        if edges_to_explore:
            return random.choice(edges_to_explore)
        # TODO Get route to the nearest unexplored node, return the label of the edge that points to it.
        return random.choice(explored_graph.current_state.get_edge_labels())

    # We receive the name of the first node when we connect, so it needs special treatment.
    first_node_name = access_function.connect().strip()
    explored_graph.current_state = explored_graph.add_node(
        ExploredGraphNode(name=first_node_name, edges=[])
    )
    # This is currently not used. The idea is to track the current path, then when we loop through the end back to
    # start, to extract "known paths" (from A to B etc) from it, then use them as shortcuts for the cases where we
    # reach an explored node and are trying to figure out how to get to the neares unexplored node from it.
    # For now though, we can leave that exercise to the astute reader.
    explored_graph.current_path = ExploredGraphPath(
        from_node=explored_graph.current_state,
        to_node=explored_graph.current_state,
        route=[],
    )

    while number_steps < MAX_EXPECTED_STEPS:
        if number_steps > 0 and explored_graph.is_fully_explored():
            # If number_steps is 0, then we have no nodes so none of them is unexplored...
            break

        number_steps += 1

        next_edge_label = get_next_edge_label()
        next_node_name = access_function.query_current_node(
            next_edge_label
        ).strip()  # Remove the new line at the end.
        if "\n" in next_node_name:
            # This is a special case, we got two transitions at once: "Z\nA".
            # This means we reached the final state, and are being returned to start.
            received_names = next_node_name.split("\n")
            last_node = explored_graph.get_node_by_name(received_names[0])
            if not last_node:
                # This is the end node. We may be visiting it for the first time, and we need to create a record for
                # it if that is the case.
                last_node = explored_graph.add_node(
                    ExploredGraphNode(name=received_names[0], edges=[])
                )
            last_edge = explored_graph.current_state.add_edge(
                ExploredGraphNode.ExploredGraphEdge(
                    to_node=last_node, label=next_edge_label
                )
            )
            # The second part of the transition is back to the first node, so it should exist already in the list.
            first_node = explored_graph.get_node_by_name(received_names[1])
            last_node.add_edge(
                ExploredGraphNode.ExploredGraphEdge(to_node=first_node, label=None)
            )
            # Save the path from start to end in the list of known paths.
            explored_graph.current_path.route.append(last_edge)
            explored_graph.current_path.to_node = last_node
            explored_graph.add_known_path(explored_graph.current_path)

            # Start the new pass.
            explored_graph.current_state = first_node
            number_passes += 1

            # Initialise a new path for the next pass.
            explored_graph.current_path = ExploredGraphPath(
                from_node=first_node,
                to_node=first_node,
                route=[],
            )

            continue

        # This is the normal case - we move from one "regular" (not final) node to another.
        next_node = explored_graph.get_node_by_name(next_node_name)
        if next_node is None:
            # If it is not None, we have been here before - e.g. reached if by another path.
            next_node = explored_graph.add_node(
                ExploredGraphNode(name=next_node_name, edges=[])
            )

        # This is either a new edge, or an existing one which we "travelled" again -
        # in which case "add_edge" will not create a duplicate but will return the existing edge.
        travelled_edge = explored_graph.current_state.add_edge(
            ExploredGraphNode.ExploredGraphEdge(
                to_node=next_node, label=next_edge_label
            )
        )
        explored_graph.current_state = next_node
        explored_graph.current_path.to_node = next_node
        explored_graph.current_path.route.append(travelled_edge)

    print(
        f"Finished in {number_steps} steps in {number_passes} passes, "
        f"discovered {len(explored_graph.nodes)} nodes and {explored_graph.get_total_edge_count()} edges."
    )
    explored_graph.visualize()
    return explored_graph


def run_exploration():
    access_function = AccessFunction()
    explore_graph(access_function=access_function)
