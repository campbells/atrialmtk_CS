"""Command-line helper for fitting Universal Atrial Coordinates.

The tool extracts a light-weight subset of the original pipeline so users
can generate the Laplace boundary files required to fit the atrial
Universal Coordinate System (UAC) to a supplied left atrial geometry.
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

from .core_utils import (
    boundary_markers,
    closest_node,
    extract_subsurface,
    find_index,
    find_junction_nodes,
    find_mitral_valve_nodes,
    geodesic_between_points,
    geodesic_midpoint_on_boundary,
    load_landmarks,
    read_mesh,
    write_vtx_extra,
)


DEFAULT_LABELS = {
    "la": 11,
    "lspv": 21,
    "lipv": 23,
    "rspv": 25,
    "ripv": 27,
}


def _with_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def fit_left_atrial_coordinates(
    geometry: str,
    mesh_name: str = "Labelled",
    seed_file: str = "seedsfileOUT_Landmarks.vtk",
    scale_factor: float = 1.0,
    labels: dict[str, int] | None = None,
    output_prefix: str | None = None,
    increment: int = 1,
) -> dict[str, np.ndarray]:
    """Compute UAC boundary constraints for a left atrial mesh.

    Parameters
    ----------
    geometry: path containing the CARP-format ``mesh_name`` files or a
        standalone XML surface mesh.
    mesh_name: mesh prefix (without extension).
    seed_file: landmark file describing PV markers in physical space.
    scale_factor: optional scaling applied when reading the landmark file.
    labels: optional override for region labels. Keys: ``la``, ``lspv``,
        ``lipv``, ``rspv``, ``ripv``.
    output_prefix: optional prefix for the VTX outputs. Defaults to
        ``geometry_dir``.
    increment: stride when sampling PV boundary nodes for geodesics.

    Returns
    -------
    dict mapping output names to the underlying node indices.
    """

    geometry_path = os.path.abspath(geometry)
    is_file = os.path.isfile(geometry_path)
    base_dir = _with_sep(os.path.dirname(geometry_path) if is_file else geometry_path)
    output_prefix = base_dir if output_prefix is None else _with_sep(output_prefix)
    label_cfg = {**DEFAULT_LABELS, **(labels or {})}

    seed_path = seed_file if os.path.isabs(seed_file) else os.path.join(base_dir, seed_file)
    pts_src = load_landmarks(seed_path, scale_factor)
    pts, elems, tot_pts, surface = read_mesh(geometry_path if is_file else base_dir, mesh_name)

    nodes_lspv = find_junction_nodes(elems, label_cfg["la"], label_cfg["lspv"])
    nodes_rspv = find_junction_nodes(elems, label_cfg["la"], label_cfg["rspv"])
    nodes_lipv = find_junction_nodes(elems, label_cfg["la"], label_cfg["lipv"])
    nodes_ripv = find_junction_nodes(elems, label_cfg["la"], label_cfg["ripv"])

    nodes_mv = find_mitral_valve_nodes(elems)

    tot_pts_surface, surface_filter = tot_pts, surface

    _, lipv_lspv_2 = boundary_markers(nodes_lipv, nodes_lspv, tot_pts_surface)
    _, ripv_rspv_2 = boundary_markers(nodes_ripv, nodes_rspv, tot_pts_surface)

    pts_lspv = pts[nodes_lspv]
    p1 = pts[nodes_lspv[lipv_lspv_2]]

    pointd = ((pts_lspv - p1) ** 2).sum(1)
    found_index = find_index(pointd, max)

    surface_filter_sub_lspv = extract_subsurface(base_dir, pts, elems, nodes_lspv)
    p2 = pts[nodes_lspv[found_index]]
    path_pts, _ = geodesic_between_points(surface_filter_sub_lspv, p1, p2, tot_pts_surface)
    mid_point = path_pts[len(path_pts) // 2]

    index = geodesic_midpoint_on_boundary(surface_filter_sub_lspv, mid_point, nodes_lspv, increment, tot_pts_surface)
    mp = pts[index]
    tpts1, _ = geodesic_between_points(surface_filter_sub_lspv, p1, mp, tot_pts_surface)
    tpts2, _ = geodesic_between_points(surface_filter_sub_lspv, mp, p2, tot_pts_surface)
    alt_path = np.concatenate([tpts1, tpts2])

    lspv_marker_pts = pts_src[2]
    m1 = ((path_pts - lspv_marker_pts) ** 2).sum(1).min()
    m2 = ((alt_path - lspv_marker_pts) ** 2).sum(1).min()

    path_candidates = [path_pts, alt_path]
    lspv_indices_path: list[int] = []
    lspv_indices_at: list[int] = []
    d = np.empty(len(path_candidates), float)

    for ind in range(len(pts_lspv)):
        for i in range(len(path_candidates)):
            index = closest_node(pts_lspv[ind], path_candidates[i]).argmin()
            d[i] = ((pts_lspv[ind] - path_candidates[i][index]) ** 2).sum()

        if d[0] <= d[1]:
            lspv_indices_path.append(nodes_lspv[ind])
            if d[0] == d[1]:
                lspv_indices_at.append(nodes_lspv[ind])
        else:
            lspv_indices_at.append(nodes_lspv[ind])

    path_lspv = lspv_indices_path if m1 < m2 else lspv_indices_at

    p3 = pts[nodes_rspv[ripv_rspv_2]]
    pts_rspv = pts[nodes_rspv]
    pointd = ((pts_rspv - p3) ** 2).sum(1)
    found_index = find_index(pointd, max)
    p4 = pts[nodes_rspv[found_index]]

    surface_filter_sub_rspv = extract_subsurface(base_dir, pts, elems, nodes_rspv)
    path_pts, _ = geodesic_between_points(surface_filter_sub_rspv, p3, p4, tot_pts_surface)
    mid_point = path_pts[len(path_pts) // 2]

    found_index = geodesic_midpoint_on_boundary(surface_filter_sub_rspv, mid_point, nodes_rspv, increment, tot_pts_surface)
    mp = tot_pts_surface[found_index]
    tpts1, _ = geodesic_between_points(surface_filter_sub_rspv, p3, mp, tot_pts_surface)
    tpts2, _ = geodesic_between_points(surface_filter_sub_rspv, mp, p4, tot_pts_surface)
    alt_path = np.concatenate([tpts1, tpts2])

    rspv_marker_pts = pts_src[3]
    m3 = ((path_pts - rspv_marker_pts) ** 2).sum(1).min()
    m4 = ((alt_path - rspv_marker_pts) ** 2).sum(1).min()

    path_candidates = [path_pts, alt_path]
    rspv_indices_path: list[int] = []
    rspv_indices_at: list[int] = []
    d = np.empty(len(path_candidates), float)

    for ind in range(len(pts_rspv)):
        for i in range(len(path_candidates)):
            index = closest_node(pts_rspv[ind], path_candidates[i]).argmin()
            d[i] = ((pts_rspv[ind] - path_candidates[i][index]) ** 2).sum()

        if d[0] <= d[1]:
            rspv_indices_path.append(nodes_rspv[ind])
            if d[0] == d[1]:
                rspv_indices_at.append(nodes_rspv[ind])
        else:
            rspv_indices_at.append(nodes_rspv[ind])

    path_rspv = rspv_indices_path if m3 < m4 else rspv_indices_at

    _, path_roof_inds = geodesic_between_points(surface_filter, p2, p4, tot_pts_surface)
    total_roof_inds = np.concatenate([path_lspv, path_roof_inds, path_rspv])

    pa_path = os.path.join(output_prefix, "PAbc1.vtx")
    lr_path = os.path.join(output_prefix, "PAbc2.vtx")
    write_vtx_extra(nodes_mv, pa_path)
    write_vtx_extra(total_roof_inds, lr_path)

    lspv_marker = pts_src[0]
    rspv_marker = pts_src[1]

    _, path_lspv_inds = geodesic_between_points(surface_filter, lspv_marker, p1, tot_pts_surface)
    _, path_rspv_inds = geodesic_between_points(surface_filter, rspv_marker, p3, tot_pts_surface)

    return {
        "mv_boundary": nodes_mv,
        "roof_boundary": total_roof_inds,
        "pa_file": pa_path,
        "lr_file": lr_path,
        "lspv_path": path_lspv_inds,
        "rspv_path": path_rspv_inds,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit Universal Atrial Coordinates to a left atrial mesh")
    parser.add_argument("geometry", help="Path to a CARP mesh directory or an XML surface mesh file")
    parser.add_argument("--mesh-name", default="Labelled", help="Mesh prefix (default: Labelled)")
    parser.add_argument("--seed-file", default="seedsfileOUT_Landmarks.vtk", help="Landmark file used for PV markers")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Optional scaling applied to landmarks")
    parser.add_argument("--la-label", type=int, default=DEFAULT_LABELS["la"], help="Left atrium label")
    parser.add_argument("--lspv-label", type=int, default=DEFAULT_LABELS["lspv"], help="Left superior pulmonary vein label")
    parser.add_argument("--lipv-label", type=int, default=DEFAULT_LABELS["lipv"], help="Left inferior pulmonary vein label")
    parser.add_argument("--rspv-label", type=int, default=DEFAULT_LABELS["rspv"], help="Right superior pulmonary vein label")
    parser.add_argument("--ripv-label", type=int, default=DEFAULT_LABELS["ripv"], help="Right inferior pulmonary vein label")
    parser.add_argument("--increment", type=int, default=1, help="Stride for sampling PV rim geodesics")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional directory/prefix where PAbc*.vtx files are written (defaults to the geometry directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    label_cfg = {
        "la": args.la_label,
        "lspv": args.lspv_label,
        "lipv": args.lipv_label,
        "rspv": args.rspv_label,
        "ripv": args.ripv_label,
    }

    fit_left_atrial_coordinates(
        geometry=args.geometry,
        mesh_name=args.mesh_name,
        seed_file=args.seed_file,
        scale_factor=args.scale_factor,
        labels=label_cfg,
        output_prefix=args.output_prefix,
        increment=args.increment,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
