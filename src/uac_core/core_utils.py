"""Minimal utilities required for fitting Universal Atrial Coordinates.

The functions here are extracted from the full `uac` package and
pared down so they can be used without the rest of the pipeline.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import vtk


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
# Map CARP to VTK element types
VTK_ELEMENT_TYPES = {"Ln": 3, "Tr": 5, "Qd": 9, "Tt": 10, "Py": 14, "Pr": 13, "Hx": 11}


def find_index(values: Sequence[float], selector) -> int:
    target = selector(values)
    for idx, value in enumerate(values):
        if value == target:
            return idx
    return 0


def closest_node(node: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    deltas = nodes - node
    return np.einsum("ij,ij->i", deltas, deltas)


def load_landmarks(filepath: str, scale_factor: float = 1.0) -> np.ndarray:
    if filepath.endswith(".vtk"):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        polydata = reader.GetOutput()
        size = polydata.GetNumberOfPoints()
        points = np.empty((size, 3), float)
        for i in range(size):
            points[i] = np.array(polydata.GetPoint(i))
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


def write_vtk(pts: np.ndarray, elems: Sequence[Element], outmesh: str) -> None:
    with open(outmesh, "w") as outfile:
        outfile.write("# vtk DataFile Version 2.0\n")
        outfile.write("converted from a CARP mesh\n")
        outfile.write("ASCII\n")
        outfile.write("DATASET UNSTRUCTURED_GRID\n")
        outfile.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            outfile.write(f"{p[0]} {p[1]} {p[2]}\n")
        outfile.write("\n")

        numcells = sum(ELEMINFOS[e.type] + 1 for e in elems)
        outfile.write(f"CELLS {len(elems)} {numcells}\n")
        for e in elems:
            outfile.write(str(ELEMINFOS[e.type]) + " " + " ".join(map(str, e.n)) + "\n")
        outfile.write("\n")

        outfile.write(f"CELL_TYPES {len(elems)}\n")
        for e in elems:
            outfile.write(str(VTK_ELEMENT_TYPES[e.type]) + "\n")
        outfile.write("\n")

        outfile.write(f"CELL_DATA {len(elems)}\n")
        outfile.write("SCALARS region double\n")
        outfile.write("LOOKUP_TABLE default\n")
        for e in elems:
            outfile.write(f"{e.reg}\n")
        outfile.write("\n")


def convert_unstructureddata_to_polydata(surface: vtk.vtkUnstructuredGrid) -> vtk.vtkPolyData:
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(surface)
    geom.Update()
    return geom.GetOutput()


def build_surface_filter_from_polydata(polydata: vtk.vtkPolyData) -> Tuple[np.ndarray, vtk.vtkDataSetSurfaceFilter]:
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(polydata)
    surface_filter.Update()
    tot_pts, _ = read_vtk_surface(polydata)
    return tot_pts, surface_filter


def clean_polydata(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    return cleaner.GetOutput()


def get_vtk_from_file(filename: str) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    return reader.GetOutput()


def read_vtk_surface(surface: vtk.vtkPolyData) -> Tuple[np.ndarray, vtk.vtkDataSetSurfaceFilter]:
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(surface)
    surface_filter.Update()
    polydata = surface_filter.GetOutput()

    size = polydata.GetNumberOfPoints()
    tot_pts = np.empty((size, 3), float)
    for i in range(size):
        tot_pts[i] = np.array(polydata.GetPoint(i))
    return tot_pts, surface_filter


def _load_vtk_xml_mesh(filename: str) -> Tuple[np.ndarray, List[Element], np.ndarray, vtk.vtkDataSetSurfaceFilter]:
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    if output is None:
        raise ValueError(f"Unable to read mesh from {filename}")

    if isinstance(output, vtk.vtkUnstructuredGrid):
        surface = convert_unstructureddata_to_polydata(output)
    elif isinstance(output, vtk.vtkPolyData):
        surface = output
    else:
        raise ValueError(f"Unsupported VTK XML mesh type for {filename}: {type(output)}")

    cell_data = surface.GetCellData()
    reg_array = cell_data.GetArray("region")
    if reg_array is None:
        reg_array = cell_data.GetArray("RegionId")

    elems: List[Element] = []
    surface.BuildCells()
    cell_points = surface.GetPolys() if surface.GetPolys() else surface.GetCells()
    cell_points.InitTraversal()
    id_list = vtk.vtkIdList()
    while cell_points.GetNextCell(id_list):
        npts = id_list.GetNumberOfIds()
        carp_type = CARP_SURFACE_TYPES.get(npts)
        if carp_type is None:
            continue
        nodes = [id_list.GetId(i) for i in range(npts)]
        reg_val = reg_array.GetTuple1(len(elems)) if reg_array is not None else 0
        elems.append(Element(carp_type, nodes, reg_val))

    tot_pts, surface_filter = read_vtk_surface(surface)
    return tot_pts, elems, tot_pts, surface_filter


def _load_fenics_xml_mesh(filename: str) -> Tuple[np.ndarray, List[Element], np.ndarray, vtk.vtkDataSetSurfaceFilter]:
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

    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    for p in pts:
        vtk_points.InsertNextPoint(*p)
    polydata.SetPoints(vtk_points)

    polys = vtk.vtkCellArray()
    for elem in elems:
        id_list = vtk.vtkIdList()
        id_list.SetNumberOfIds(len(elem.n))
        for i, nid in enumerate(elem.n):
            id_list.SetId(i, int(nid))
        polys.InsertNextCell(id_list)
    polydata.SetPolys(polys)

    tot_pts, surface_filter = build_surface_filter_from_polydata(polydata)
    return pts, elems, tot_pts, surface_filter


def read_xml_mesh(filename: str) -> Tuple[np.ndarray, List[Element], np.ndarray, vtk.vtkDataSetSurfaceFilter]:
    """Read an atrial surface mesh stored as an XML file.

    The loader supports VTK XML PolyData/UnstructuredGrid files as well as
    the DOLFIN/FEniCS legacy XML mesh layout.
    """

    try:
        return _load_vtk_xml_mesh(filename)
    except Exception:
        return _load_fenics_xml_mesh(filename)


def read_carp_mesh(base_dir: str, mesh_name: str) -> Tuple[np.ndarray, List[Element], np.ndarray, vtk.vtkPolyData]:
    mesh_prefix = os.path.join(base_dir, mesh_name)
    pts = read_pts(mesh_prefix)
    elems = read_elem(mesh_prefix)

    tmp_file = os.path.join(base_dir, "tmp_uac_core.vtk")
    write_vtk(pts, elems, tmp_file)
    surface = get_vtk_from_file(tmp_file)
    os.remove(tmp_file)
    surface = convert_unstructureddata_to_polydata(surface)
    tot_pts, surface_filter = read_vtk_surface(surface)
    return pts, elems, tot_pts, surface_filter


def read_mesh(geometry: str, mesh_name: str | None = None):
    """Read a mesh from either CARP components or an XML surface."""

    if os.path.isfile(geometry):
        ext = os.path.splitext(geometry)[1].lower()
        if ext == ".xml":
            return read_xml_mesh(geometry)
        if ext == ".vtk":
            surface = get_vtk_from_file(geometry)
            surface = convert_unstructureddata_to_polydata(surface)
            tot_pts, surface_filter = read_vtk_surface(surface)
            elems = []
            return tot_pts, elems, tot_pts, surface_filter
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


def geodesic_between_points(
    surface_filter: vtk.vtkDataSetSurfaceFilter, start: np.ndarray, end: np.ndarray, tot_pts: np.ndarray
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

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface_filter.GetOutput())
    dijkstra.SetStartVertex(lspv)
    dijkstra.SetEndVertex(rspv)
    dijkstra.StopWhenEndReachedOn()
    dijkstra.UseScalarWeightsOff()
    dijkstra.Update()

    id_len = dijkstra.GetIdList().GetNumberOfIds()
    indices = np.empty(id_len, int)
    mpts = np.empty((id_len, 3), float)
    for i in range(id_len):
        indices[i] = dijkstra.GetIdList().GetId(id_len - i - 1)
        mpts[i] = tot_pts[indices[i]]
    return mpts, indices


def geodesic_midpoint_on_boundary(
    surface_filter: vtk.vtkDataSetSurfaceFilter, mid_point: np.ndarray, nodes_la_pv: np.ndarray, increment: int, tot_pts: np.ndarray
) -> int:
    pointd = ((tot_pts - mid_point) ** 2).sum(1)
    index = find_index(pointd, min)

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface_filter.GetOutput())
    dijkstra.StopWhenEndReachedOn()
    dijkstra.UseScalarWeightsOff()
    dijkstra.SetEndVertex(index)

    len_store = []
    for ep in range(0, len(nodes_la_pv), increment):
        dijkstra.SetStartVertex(nodes_la_pv[ep])
        dijkstra.Update()
        len_store.append(dijkstra.GetIdList().GetNumberOfIds())

    index = find_index(len_store, max)
    index = list(range(0, len(nodes_la_pv), increment))[index]
    found_index = nodes_la_pv[index]
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
    tmp_file = os.path.join(base_dir, "tmp_uac_core_subsurface.vtk")
    write_vtk(pts, elems_del, tmp_file)
    surface = get_vtk_from_file(tmp_file)
    os.remove(tmp_file)
    surface = convert_unstructureddata_to_polydata(surface)
    if return_pts:
        tot_pts, _ = read_vtk_surface(surface)
        return tot_pts
    return surface
