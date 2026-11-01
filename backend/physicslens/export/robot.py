"""URDF and SDF exports of a verified pendulum, spring or projectile, next to the MuJoCo one.

Each format carries what it can, and says in the file what it cannot. URDF has no world (no gravity) and no joint
springs, so a spring's stiffness and the measured g are written as comments for the simulator's own settings; SDF has
both. Drag on a free projectile is in neither, so the comment gives the fitted coefficient. Every file is checked by
parsing it back (export_xml), never written blind.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from ..physics.custom import spec_for
from .mjcf import NOMINAL_MASS

SUPPORTED = ("pendulum", "oscillator", "projectile")


class RobotExportError(ValueError):
    pass


def _check(xml: str, root_tag: str, required: tuple[str, ...]) -> str:
    """Parse the file back and make sure the elements a simulator needs are there."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise RobotExportError(f"generated {root_tag} does not parse: {e}") from e
    if root.tag != root_tag:
        raise RobotExportError(f"generated file's root is <{root.tag}>, not <{root_tag}>")
    missing = [path for path in required if root.find(path) is None]
    if missing:
        raise RobotExportError(f"generated {root_tag} lacks {missing}")
    return xml


def _inertia(mass: float) -> str:
    i = 2 / 5 * mass * 0.02**2  # a 2 cm ball: small, nonzero (simulators reject a zero inertia)
    return f'<inertia ixx="{i:.6g}" ixy="0" ixz="0" iyy="{i:.6g}" iyz="0" izz="{i:.6g}"/>'


def urdf(fit_state: dict[str, Any], title: str = "PhysicsLens model") -> str:
    spec = spec_for(fit_state)
    p, ctx = fit_state["params"], fit_state.get("ctx") or {}
    m = NOMINAL_MASS
    notes = [f"{title}: {spec.title}, identified by PhysicsLens and verified on held-out motion."]
    if spec.family == "pendulum":
        L = float(ctx["length_m"])
        damping = float(p.get("gamma", 0.0)) * m * L * L  # joint torque per angular speed
        if "beta" in p:
            notes.append(f"Air drag beta = {p['beta']:.4g}/rad has no URDF equivalent; use the Python export for it.")
        notes.append(f"Set the simulator's gravity to {p['g']:.5g} m/s^2 (URDF has no world).")
        body = (f'  <link name="pivot"/>\n'
                f'  <link name="bob">\n    <inertial><origin xyz="0 0 {-L:.6g}"/><mass value="{m}"/>{_inertia(m)}</inertial>\n'
                f'    <visual><origin xyz="0 0 {-L:.6g}"/><geometry><sphere radius="0.02"/></geometry></visual>\n  </link>\n'
                f'  <joint name="swing" type="continuous">\n    <parent link="pivot"/><child link="bob"/>\n'
                f'    <axis xyz="0 -1 0"/>\n    <dynamics damping="{damping:.6g}" friction="0"/>\n  </joint>\n')
    elif spec.family == "oscillator":
        k = m * float(p["omega"]) ** 2
        c = 2 * m * float(p.get("zeta", 0.0)) * float(p["omega"])
        notes.append(f"URDF has no joint spring: stiffness {k:.6g} N/m (mass {m} kg), rest position {p['s0']:.4g} m. "
                     "Add it in the simulator (Gazebo: <spring_stiffness> in the SDF export).")
        if "f" in p:
            notes.append(f"Dry friction f = {p['f']:.4g} m/s^2 becomes joint friction {m * p['f']:.4g} N.")
        friction = m * float(p.get("f", 0.0))
        body = (f'  <link name="base"/>\n'
                f'  <link name="mass">\n    <inertial><mass value="{m}"/>{_inertia(m)}</inertial>\n'
                f'    <visual><geometry><sphere radius="0.02"/></geometry></visual>\n  </link>\n'
                f'  <joint name="slide" type="prismatic">\n    <parent link="base"/><child link="mass"/>\n'
                f'    <axis xyz="0 0 1"/>\n    <limit lower="-10" upper="10" effort="1000" velocity="100"/>\n'
                f'    <dynamics damping="{c:.6g}" friction="{friction:.6g}"/>\n  </joint>\n')
    elif spec.family == "projectile":
        notes.append(f"Set the simulator's gravity to {p['g']:.5g} m/s^2 (URDF has no world).")
        for name, what in (("k", "linear drag k"), ("c", "quadratic drag c")):
            if name in p:
                notes.append(f"Fitted {what} = {p[name]:.4g} has no URDF equivalent; apply it as a force, or use the "
                             "Python export.")
        body = (f'  <link name="ball">\n    <inertial><mass value="{m}"/>{_inertia(m)}</inertial>\n'
                f'    <visual><geometry><sphere radius="0.02"/></geometry></visual>\n'
                f'    <collision><geometry><sphere radius="0.02"/></geometry></collision>\n  </link>\n')
    else:
        raise RobotExportError(f"URDF export covers {', '.join(SUPPORTED)}; this is a {spec.family}.")
    comment = "\n".join(f"  <!-- {n.replace('--', '-')} -->" for n in notes)
    xml = f'<?xml version="1.0"?>\n<robot name="physicslens_{spec.family}">\n{comment}\n{body}</robot>\n'
    return _check(xml, "robot", ("link",))


def sdf(fit_state: dict[str, Any], title: str = "PhysicsLens model") -> str:
    spec = spec_for(fit_state)
    p, ctx = fit_state["params"], fit_state.get("ctx") or {}
    m = NOMINAL_MASS
    g = float(p.get("g", 9.81))
    notes = [f"{title}: {spec.title}, identified by PhysicsLens and verified on held-out motion."]
    inertial = f"<inertial><mass>{m}</mass><inertia>" + "".join(
        f"<{a}>{v:.6g}</{a}>" for a, v in (("ixx", 2 / 5 * m * 0.0004), ("iyy", 2 / 5 * m * 0.0004), ("izz", 2 / 5 * m * 0.0004),
                                          ("ixy", 0), ("ixz", 0), ("iyz", 0))) + "</inertia></inertial>"
    sphere = "<geometry><sphere><radius>0.02</radius></sphere></geometry>"
    if spec.family == "pendulum":
        L = float(ctx["length_m"])
        damping = float(p.get("gamma", 0.0)) * m * L * L
        if "beta" in p:
            notes.append(f"Air drag beta = {p['beta']:.4g}/rad has no SDF joint equivalent; use the Python export.")
        model = (f'      <link name="pivot"/>\n'
                 f'      <link name="bob"><pose>0 0 {-L:.6g} 0 0 0</pose>{inertial}'
                 f'<visual name="v">{sphere}</visual></link>\n'
                 f'      <joint name="swing" type="revolute"><parent>pivot</parent><child>bob</child>'
                 f'<pose relative_to="pivot">0 0 0 0 0 0</pose>'
                 f'<axis><xyz>0 -1 0</xyz><dynamics><damping>{damping:.6g}</damping></dynamics></axis></joint>\n'
                 f'      <joint name="fix" type="fixed"><parent>world</parent><child>pivot</child></joint>\n')
    elif spec.family == "oscillator":
        k = m * float(p["omega"]) ** 2
        c = 2 * m * float(p.get("zeta", 0.0)) * float(p["omega"])
        friction = m * float(p.get("f", 0.0))
        model = (f'      <link name="base"/>\n'
                 f'      <link name="mass">{inertial}<visual name="v">{sphere}</visual></link>\n'
                 f'      <joint name="slide" type="prismatic"><parent>base</parent><child>mass</child>'
                 f'<axis><xyz>0 0 1</xyz><limit><lower>-10</lower><upper>10</upper></limit>'
                 f'<dynamics><damping>{c:.6g}</damping><friction>{friction:.6g}</friction>'
                 f'<spring_reference>{float(p["s0"]):.6g}</spring_reference><spring_stiffness>{k:.6g}</spring_stiffness>'
                 f'</dynamics></axis></joint>\n'
                 f'      <joint name="fix" type="fixed"><parent>world</parent><child>base</child></joint>\n')
    elif spec.family == "projectile":
        for name, what in (("k", "linear drag k"), ("c", "quadratic drag c")):
            if name in p:
                notes.append(f"Fitted {what} = {p[name]:.4g} is not modelled here; apply it as a force plugin, or use the "
                             "Python export.")
        model = f'      <link name="ball">{inertial}<visual name="v">{sphere}</visual><collision name="c">{sphere}</collision></link>\n'
    else:
        raise RobotExportError(f"SDF export covers {', '.join(SUPPORTED)}; this is a {spec.family}.")
    comment = "\n".join(f"  <!-- {n.replace('--', '-')} -->" for n in notes)
    xml = (f'<?xml version="1.0"?>\n<sdf version="1.9">\n{comment}\n  <world name="physicslens">\n'
           f'    <gravity>0 0 {-g:.6g}</gravity>\n    <model name="physicslens_{spec.family}">\n{model}    </model>\n'
           f'  </world>\n</sdf>\n')
    return _check(xml, "sdf", ("world/gravity", "world/model/link"))


def gravity_of(xml: str) -> float | None:
    """The z gravity an SDF file declares (for tests and for checking a round trip)."""
    g = ET.fromstring(xml).find("world/gravity")
    return None if g is None or not g.text else -float(g.text.split()[2])


__all__ = ["SUPPORTED", "RobotExportError", "gravity_of", "sdf", "urdf"]
