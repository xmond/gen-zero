"""Deterministic Graph AST and Layout Mutation Engine.

Phase 4 of the Sandwich Streaming Pipeline:
- Strictly prohibits LLMs from directly generating fragile raw XML or freeform coordinates.
- Programmatically executes GraphMutation operations on an in-memory DAG.
- Emits deterministic, collision-free Draw.io XML (mxGraph format) and Mermaid flowchart code.
- Fully supports human-in-the-loop editing with transparent draft overlays.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import html
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET


class GraphOp(str, Enum):
    """Atomic graph mutations."""
    CREATE_NODE = "create_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    ADD_EDGE = "add_edge"


@dataclass
class GraphMutation:
    """Explicit typed instruction for modifying the workflow graph."""
    op: GraphOp
    target_node_id: Optional[str]
    label: str
    role_lane: str
    depends_on: List[str]
    is_draft: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["op"] = self.op.value
        return d


@dataclass
class GraphNode:
    """A workflow node entity."""
    node_id: str
    label: str
    role_lane: str
    is_draft: bool = False
    x: float = 0.0
    y: float = 0.0
    width: float = 140.0
    height: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """A directed causal dependency between workflow nodes."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphAST:
    """Deterministic in-memory Directed Acyclic Graph AST and layout generator."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._node_counter = 0
        self._edge_counter = 0

    def apply_mutation(self, mutation: GraphMutation) -> str:
        """Applies mutation to graph AST and recalculates layout coordinates."""
        if mutation.op == GraphOp.CREATE_NODE:
            self._node_counter += 1
            node_id = mutation.target_node_id or f"node_{self._node_counter}"
            self.nodes[node_id] = GraphNode(
                node_id=node_id,
                label=mutation.label,
                role_lane=mutation.role_lane,
                is_draft=mutation.is_draft,
            )
            # Add dependency edges
            for parent_id in mutation.depends_on:
                if parent_id in self.nodes:
                    self._edge_counter += 1
                    self.edges.append(
                        GraphEdge(
                            edge_id=f"edge_{self._edge_counter}",
                            source_node_id=parent_id,
                            target_node_id=node_id,
                        )
                    )
            self._recompute_layout()
            return node_id

        elif mutation.op == GraphOp.UPDATE_NODE:
            target_id = mutation.target_node_id
            if target_id and target_id in self.nodes:
                node = self.nodes[target_id]
                if mutation.label:
                    node.label = mutation.label
                if mutation.role_lane:
                    node.role_lane = mutation.role_lane
                node.is_draft = mutation.is_draft
                for parent_id in mutation.depends_on:
                    if parent_id in self.nodes and not any(
                        e.source_node_id == parent_id and e.target_node_id == target_id
                        for e in self.edges
                    ):
                        self._edge_counter += 1
                        self.edges.append(
                            GraphEdge(
                                edge_id=f"edge_{self._edge_counter}",
                                source_node_id=parent_id,
                                target_node_id=target_id,
                            )
                        )
                self._recompute_layout()
                return target_id
            return ""

        elif mutation.op == GraphOp.DELETE_NODE:
            target_id = mutation.target_node_id
            if target_id and target_id in self.nodes:
                del self.nodes[target_id]
                self.edges = [
                    e for e in self.edges
                    if e.source_node_id != target_id and e.target_node_id != target_id
                ]
                self._recompute_layout()
                return target_id
            return ""

        elif mutation.op == GraphOp.ADD_EDGE:
            if mutation.depends_on and mutation.target_node_id:
                for parent_id in mutation.depends_on:
                    if parent_id in self.nodes and mutation.target_node_id in self.nodes:
                        self._edge_counter += 1
                        self.edges.append(
                            GraphEdge(
                                edge_id=f"edge_{self._edge_counter}",
                                source_node_id=parent_id,
                                target_node_id=mutation.target_node_id,
                            )
                        )
                self._recompute_layout()
                return mutation.target_node_id
            return ""

        return ""

    def _recompute_layout(self) -> None:
        """Deterministic layered topological layout."""
        if not self.nodes:
            return

        # Assign vertical levels via in-degree BFS / longest path
        in_degrees: Dict[str, int] = {node_id: 0 for node_id in self.nodes}
        adj: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            if edge.source_node_id in adj and edge.target_node_id in in_degrees:
                adj[edge.source_node_id].append(edge.target_node_id)
                in_degrees[edge.target_node_id] += 1

        levels: Dict[str, int] = {}
        queue = [n for n, deg in in_degrees.items() if deg == 0]
        for n in queue:
            levels[n] = 0

        while queue:
            curr = queue.pop(0)
            curr_lvl = levels[curr]
            for neighbor in adj.get(curr, []):
                levels[neighbor] = max(levels.get(neighbor, 0), curr_lvl + 1)
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

        # Handle any remaining nodes in cycles
        for n in self.nodes:
            if n not in levels:
                levels[n] = 0

        # Group by level
        level_groups: Dict[int, List[str]] = {}
        for n, lvl in levels.items():
            level_groups.setdefault(lvl, []).append(n)

        # Assign coordinates: x spaced by level, y spaced by rank in level
        for lvl, group in level_groups.items():
            for idx, node_id in enumerate(group):
                node = self.nodes[node_id]
                node.x = 80.0 + idx * 200.0
                node.y = 60.0 + lvl * 120.0

    def to_mermaid(self) -> str:
        """Generates standard Mermaid flowchart TD code."""
        lines = ["flowchart TD"]
        for node in self.nodes.values():
            draft_tag = " [Draft?]" if node.is_draft else ""
            clean_label = f"{node.role_lane}: {node.label}{draft_tag}"
            clean_label = clean_label.replace('"', "'")
            lines.append(f'    {node.node_id}["{clean_label}"]')
            if node.is_draft:
                lines.append(f"    style {node.node_id} stroke-dasharray: 5 5,fill:#f9f9f9,stroke:#999")

        for edge in self.edges:
            lines.append(f"    {edge.source_node_id} --> {edge.target_node_id}")

        return "\n".join(lines)

    def to_drawio_xml(self) -> str:
        """Emits valid, editable Draw.io / mxGraph XML document."""
        mxfile = ET.Element("mxfile", host="GenZero", modified="2026-09-20", agent="GenZeroStreamGraph")
        diagram = ET.SubElement(mxfile, "diagram", id="workflow_diag", name="Workflow Topology")
        mxGraphModel = ET.SubElement(
            diagram,
            "mxGraphModel",
            dx="1000",
            dy="1000",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            page="1",
            pageScale="1",
            pageWidth="1169",
            pageHeight="827",
        )
        root = ET.SubElement(mxGraphModel, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")

        # Add Nodes
        for node in self.nodes.values():
            style = "rounded=1;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=12;"
            if node.is_draft:
                style += "dashed=1;opacity=70;fillColor=#f5f5f5;strokeColor=#666666;"
            else:
                style += "fillColor=#dae8fc;strokeColor=#6c8ebf;"

            label_text = f"<b>[{node.role_lane}]</b><br/>{node.label}"
            if node.is_draft:
                label_text += "<br/><i>(Pending Draft)</i>"

            cell = ET.SubElement(
                root,
                "mxCell",
                id=node.node_id,
                value=label_text,
                style=style,
                vertex="1",
                parent="1",
            )
            ET.SubElement(
                cell,
                "mxGeometry",
                x=str(node.x),
                y=str(node.y),
                width=str(node.width),
                height=str(node.height),
                as_="geometry",
            )

        # Add Edges
        for edge in self.edges:
            edge_cell = ET.SubElement(
                root,
                "mxCell",
                id=edge.edge_id,
                value=edge.label or "",
                style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;",
                edge="1",
                parent="1",
                source=edge.source_node_id,
                target=edge.target_node_id,
            )
            ET.SubElement(edge_cell, "mxGeometry", relative="1", as_="geometry")

        return ET.tostring(mxfile, encoding="utf-8").decode("utf-8")
