"""Minimal utilities required for fitting Universal Atrial Coordinates.

The functions here are extracted from the full `uac` package and
pared down so they can be used without the rest of the pipeline.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import heapq


class Edge:
    def __init__(self, n0: int, n1: int):
        self.n = (n0, n1) if n0 < n1 else (n1, n0)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - convenience
        return isinstance(other, Edge) and self.n == other.n

    def __hash__(self) -> int:  # pragma: no cover - convenience
        return hash(self.n)


class Element:
    """Simple container describing a surface element."""

    def __init__(self, elem_type: str, nodes: Sequence[int], reg: int = 0):
        self.type = elem_type
        self.n = list(map(int, nodes))
        self.reg = int(reg)

    def edges(self) -> List[Edge]:
        return [Edge(self.n[i], self.n[(i + 1) % len(self.n)]) for i in range(len(self.n))]


class Line:
    """List of connected edges used to describe model boundaries."""

    def __init__(self):
        self.edge: set[Edge] = set()
        self.nodes: set[int] = set()

    def add_edge(self, edge: Edge) -> None:
        self.edge.add(edge)
        self.nodes |= set(edge.n)


# Number of vertices per element
ELEMINFOS = {"Ln": 2, "Tr": 3, "Qd": 4, "Tt": 4, "Py": 5, "Pr": 6, "Hx": 8}
# Map number of vertices to CARP surface element
CARP_SURFACE_TYPES = {2: "Ln", 3: "Tr", 4: "Qd"}
def find_index(values: Sequence[float], selector) -> int:
    target = selector(values)
    for idx, value in enumerate(values):
        if value == target:
            return idx
    return 0


def closest_node(node: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    deltas = nodes - node
    return np.einsum("ij,ij->i", deltas, deltas)


def _load_legacy_vtk_points(filepath: str) -> np.ndarray:
    """Parse a minimal legacy VTK PolyData file containing POINTS."""

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    point_header = None
    for idx, line in enumerate(lines):
        if line.startswith("POINTS"):
            point_header = (idx, int(line.split()[1]))
            break
    if point_header is None:
        raise ValueError(f"POINTS section not found in {filepath}")

    start, count = point_header
    points = np.empty((count, 3), float)
    for i in range(count):
        coords = list(map(float, lines[start + 1 + i].split()))
        if len(coords) != 3:
            raise ValueError(f"Invalid coordinate line in {filepath}: {lines[start + 1 + i]}")
        points[i] = coords
    return points


def load_landmarks(filepath: str, scale_factor: float = 1.0) -> np.ndarray:
    if filepath.endswith(".vtk"):
        points = _load_legacy_vtk_points(filepath)
    else:
        points = np.loadtxt(filepath, delimiter=",")
    return points * scale_factor


def read_pts(prefix: str) -> np.ndarray:
    with open(prefix + ".pts", "r") as file:
        lines = file.readlines()[1:]
        pts = np.empty((len(lines), 3), float)
        for i, l in enumerate(lines):
            pts[i] = list(map(float, l.split()))
    return pts


def read_element_file(filename: str) -> List[Element]:
    with open(filename, "r") as file:
        lines = file.readlines()[1:]
        elems: List[Element] = [None] * len(lines)
        for i, l in enumerate(lines):
            e = l.split()
            np_nodes = ELEMINFOS[e[0]]
            elems[i] = Element(e[0], e[1 : np_nodes + 1], e[np_nodes + 1] if len(e) > np_nodes + 1 else 0)
    return elems


def read_elem(prefix: str) -> List[Element]:
    return read_element_file(prefix + ".elem")


def _load_fenics_xml_mesh(filename: str) -> Tuple[np.ndarray, List[Element], np.ndarray, Dict[int, list]]:
    tree = ET.parse(filename)
    root = tree.getroot()

    vertices = root.find("vertices")
    cells = root.find("cells")
    if vertices is None or cells is None:
        raise ValueError("Invalid FEniCS/DOLFIN XML mesh: missing vertices or cells")

    pts = np.empty((int(vertices.attrib["size"]), 3), float)
    for v in vertices.findall("vertex"):
        idx = int(v.attrib["index"])
        pts[idx] = [float(v.attrib.get("x", 0.0)), float(v.attrib.get("y", 0.0)), float(v.attrib.get("z", 0.0))]

    elems: List[Element] = []
    for cell in cells:
        node_keys = sorted(k for k in cell.keys() if k.startswith("v"))
        nodes = [int(cell.attrib[key]) for key in node_keys]
        carp_type = CARP_SURFACE_TYPES.get(len(nodes))
        if carp_type is None:
            continue
        reg_val = int(cell.attrib.get("region", cell.attrib.get("physicalEntity", 0)))
        elems.append(Element(carp_type, nodes, reg_val))

    adjacency = build_adjacency(elems, pts)
    return pts, elems, pts, adjacency


def read_xml_mesh(filename: str) -> Tuple[np.ndarray, List[Element], np.ndarray, Dict[int, list]]:
    """Read an atrial surface mesh stored as a DOLFIN/FEniCS XML file."""

    return _load_fenics_xml_mesh(filename)


def read_carp_mesh(base_dir: str, mesh_name: str) -> Tuple[np.ndarray, List[Element], np.ndarray, Dict[int, list]]:
    mesh_prefix = os.path.join(base_dir, mesh_name)
    pts = read_pts(mesh_prefix)
    elems = read_elem(mesh_prefix)

    adjacency = build_adjacency(elems, pts)
    return pts, elems, pts, adjacency


def read_mesh(geometry: str, mesh_name: str | None = None):
    """Read a mesh from either CARP components or an XML surface."""

    if os.path.isfile(geometry):
        ext = os.path.splitext(geometry)[1].lower()
        if ext == ".xml":
            return read_xml_mesh(geometry)
        raise ValueError(f"Unsupported mesh format for {geometry}")

    if mesh_name is None:
        mesh_name = "Labelled"
    return read_carp_mesh(geometry, mesh_name)


def find_edges(elems: Sequence[Element]) -> Tuple[dict, set]:
    bedges: dict[Edge, int] = dict()
    for e in elems:
        for edge in e.edges():
            if edge in bedges:
                del bedges[edge]
            else:
                bedges[edge] = 1

    edgenode: set[int] = set()
    for edge in bedges:
        edgenode.add(edge.n[0])
        edgenode.add(edge.n[1])
    return bedges, edgenode


def coalesce(edges: Iterable[Edge]) -> List[Line]:
    def coalesce_loop(neighbours, line: Line):
        victims = set()
        for n in neighbours:
            edgelist = start.get(n)
            if edgelist:
                for edge in edgelist:
                    line.add_edge(edge)
                    victims.add(edge.n[1])
                start.pop(n, None)
            edgelist = end.get(n)
            if edgelist:
                for edge in edgelist:
                    line.add_edge(edge)
                    victims.add(edge.n[0])
                end.pop(n, None)
        return victims

    start = {}
    end = {}
    for edge in edges:
        start.setdefault(edge.n[0], set()).add(edge)
        end.setdefault(edge.n[1], set()).add(edge)

    lines: List[Line] = []
    while start:
        line = Line()
        # start at an end if possible
        start_n = None
        for n in start:
            if n not in end:
                start_n = n
                break
        if start_n is None:
            start_n = next(iter(start))
        start_edges = start.pop(start_n)
        for e in start_edges:
            line.add_edge(e)
        victims = {e.n[1] for e in start_edges}

        while victims:
            victims = coalesce_loop(victims, line)
        lines.append(line)
    return lines


def find_mitral_valve_nodes(elems: Sequence[Element], ascending: bool = False) -> np.ndarray:
    border, _ = find_edges(elems)
    lines = coalesce(border)
    line_size = np.array([len(line.nodes) for line in lines])
    ls = np.argsort(line_size)
    if not ascending:
        ls = ls[::-1]
    index = ls[0]
    return np.fromiter(lines[index].nodes, int, len(lines[index].nodes))


def find_junction_nodes(elems: Sequence[Element], reg1: int, reg2: int) -> np.ndarray:
    nodes = [set(), set()]
    for e in elems:
        if e.reg == reg1:
            nodes[0].update(e.n)
        elif e.reg == reg2:
            nodes[1].update(e.n)
    tot_nodes = nodes[0].intersection(nodes[1])
    return np.fromiter(tot_nodes, int, len(tot_nodes))


def boundary_markers(index1: np.ndarray, index2: np.ndarray, tot_pts: np.ndarray) -> Tuple[int, int]:
    indices = [index1, index2]
    points_t = [None] * 2
    for k in range(2):
        size = len(indices[k])
        points = np.empty((size, 3), float)
        for i in range(size):
            points[i] = tot_pts[indices[k][i]]
        points_t[k] = points

    distances = np.linalg.norm(points_t[0][:, None, :] - points_t[1][None, :, :], axis=2)
    indices = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    return indices[0], indices[1]


def build_adjacency(elems: Sequence[Element], pts: np.ndarray) -> Dict[int, list]:
    """Construct an adjacency list with Euclidean edge weights."""

    adjacency: Dict[int, list] = {}
    for elem in elems:
        for edge in elem.edges():
            n0, n1 = edge.n
            w = np.linalg.norm(pts[n0] - pts[n1])
            adjacency.setdefault(n0, []).append((n1, w))
            adjacency.setdefault(n1, []).append((n0, w))
    return adjacency


def _dijkstra(adjacency: Dict[int, list], start: int, end: int) -> Tuple[float, List[int]]:
    queue: list[tuple[float, int]] = [(0.0, start)]
    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}

    while queue:
        d, node = heapq.heappop(queue)
        if node == end:
            break
        if d > dist.get(node, float("inf")):
            continue
        for neighbour, weight in adjacency.get(node, []):
            nd = d + weight
            if nd < dist.get(neighbour, float("inf")):
                dist[neighbour] = nd
                prev[neighbour] = node
                heapq.heappush(queue, (nd, neighbour))

    if end not in dist:
        return float("inf"), []

    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[end], path


def geodesic_between_points(
    adjacency: Dict[int, list], start: np.ndarray, end: np.ndarray, tot_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pv_pts = [start, end]
    pts = [None] * 2
    for k in range(2):
        size = len(pv_pts[k]) if pv_pts[k].ndim > 1 else 1
        points = np.empty((size, 3), float)
        if pv_pts[k].ndim > 1:
            points[:] = pv_pts[k]
        else:
            points[0] = pv_pts[k]
        pts[k] = points

    p_start = pts[0][0]
    p_end = pts[1][0]
    pointd = ((tot_pts - p_start) ** 2).sum(1)
    lspv = find_index(pointd, min)
    pointd = ((tot_pts - p_end) ** 2).sum(1)
    rspv = find_index(pointd, min)

    _, path_indices = _dijkstra(adjacency, lspv, rspv)
    mpts = tot_pts[path_indices]
    return mpts, np.array(path_indices, dtype=int)


def geodesic_midpoint_on_boundary(
    adjacency: Dict[int, list], mid_point: np.ndarray, nodes_la_pv: np.ndarray, increment: int, tot_pts: np.ndarray
) -> int:
    pointd = ((tot_pts - mid_point) ** 2).sum(1)
    target_index = find_index(pointd, min)

    len_store = []
    candidates = list(range(0, len(nodes_la_pv), increment))
    for ep in candidates:
        _, path = _dijkstra(adjacency, nodes_la_pv[ep], target_index)
        len_store.append(len(path))

    index = find_index(len_store, max)
    found_index = nodes_la_pv[candidates[index]]
    return found_index


def write_vtx(ids: Sequence[int], filename: str, intra: bool) -> None:
    with open(filename, "w") as file:
        file.write(str(len(ids)) + "\n")
        file.write("intra\n" if intra else "extra\n")
        for d in ids:
            file.write(str(int(d)) + "\n")


def write_vtx_extra(ids: Sequence[int], filename: str) -> None:
    write_vtx(ids, filename, False)


def extract_subsurface(
    base_dir: str, pts: np.ndarray, elems: Sequence[Element], node_del: Sequence[int], return_pts: bool = False
):
    def element2delete(elems, node_del):
        node_del_set = set(node_del)
        elems_del = []
        for e in elems:
            if not node_del_set.intersection(e.n):
                elems_del.append(e)
        return elems_del

    elems_del = element2delete(elems, node_del)
    if return_pts:
        return pts
    return build_adjacency(elems_del, pts)
