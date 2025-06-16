from pydantic import BaseModel

class Node(BaseModel):
    name: str
    description: str = ""
    child_nodes: list["Node"] = []

    def get_formatted_taxonomy(self, level: int = 0) -> str:
        # Generate formatted tree with the indentation specifying how deep the node is
        lines = [f"{'   ' * level}- {self.name}: " + self.description]
        for child in self.child_nodes:
            formatted_subtree = child.get_formatted_taxonomy(level + 1)
            lines.append(formatted_subtree)
        return "\n".join(lines)

    def get_formatted_leaf_nodes(self, path: str = "") -> list[str]:
        # Generate a list of node traversals to each leaf nodes e.g. ["node1.node2.leaf_node1: description", ...]
        current_path = f"- {path}.{self.name}" if path else self.name
        current_path += f": {self.description}" if self.description else ""

        if not self.child_nodes:
            return [current_path]

        leaf_nodes = []
        for child in self.child_nodes:
            leaf_nodes += child.get_formatted_leaf_nodes(current_path)
        return leaf_nodes