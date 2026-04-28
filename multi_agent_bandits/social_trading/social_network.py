import random


class SocialNetwork:
    """
    Lightweight undirected graph used for local communication.
    """

    def __init__(self, adjacency):
        self.adjacency = {
            node: sorted(neighbors) for node, neighbors in adjacency.items()
        }

    @classmethod
    def fully_connected(cls, n_agents):
        adjacency = {}
        for agent_idx in range(n_agents):
            adjacency[agent_idx] = {
                other_idx for other_idx in range(n_agents) if other_idx != agent_idx
            }
        return cls(adjacency)

    @classmethod
    def random_graph(cls, n_agents, edge_prob=0.3, rng=None):
        rng = rng or random
        adjacency = {agent_idx: set() for agent_idx in range(n_agents)}
        for left in range(n_agents):
            for right in range(left + 1, n_agents):
                if rng.random() < edge_prob:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        return cls(adjacency)

    @classmethod
    def ring_lattice(cls, n_agents, neighbors_per_side=1):
        adjacency = {agent_idx: set() for agent_idx in range(n_agents)}
        for agent_idx in range(n_agents):
            for offset in range(1, neighbors_per_side + 1):
                left = (agent_idx - offset) % n_agents
                right = (agent_idx + offset) % n_agents
                adjacency[agent_idx].add(left)
                adjacency[agent_idx].add(right)
        return cls(adjacency)

    @classmethod
    def from_topology(
        cls,
        topology,
        n_agents,
        edge_prob=0.3,
        neighbors_per_side=1,
        rng=None,
    ):
        if topology == "fully_connected":
            return cls.fully_connected(n_agents)
        if topology == "random_graph":
            return cls.random_graph(n_agents, edge_prob=edge_prob, rng=rng)
        if topology == "ring_lattice":
            return cls.ring_lattice(n_agents, neighbors_per_side=neighbors_per_side)
        raise ValueError(f"Unknown topology: {topology}")

    def neighbors(self, agent_idx):
        return list(self.adjacency.get(agent_idx, []))

    def edge_list(self):
        edges = []
        for left, neighbors in self.adjacency.items():
            for right in neighbors:
                if left < right:
                    edges.append((left, right))
        return edges
