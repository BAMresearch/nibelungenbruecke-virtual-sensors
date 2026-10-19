"""CLI: ``python -m nibelungen_virtual_sensors.ifc {reference,sensors} ...``"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import _data, build_reference, virtual_sensors_from_ifc, write_sensor_file


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m nibelungen_virtual_sensors.ifc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="derive bridge frame + girder section from the STB IFC")
    r.add_argument("structure_ifc")
    r.add_argument("--out-dir", default=None, help="default: the package data folder")
    s = sub.add_parser("sensors", help="map the sensor proxies of an IFC into the model")
    s.add_argument("sensor_ifc")
    s.add_argument("-o", "--output", default=None,
                   help="JSON file (default: the packaged data/virtual_sensors.json)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "reference":
        f = build_reference(a.structure_ifc, a.out_dir)
        print(json.dumps(f.__dict__, indent=2))
    else:
        sensors = virtual_sensors_from_ifc(a.sensor_ifc)
        out = a.output or str(_data("virtual_sensors.json"))
        write_sensor_file(sensors, out, Path(a.sensor_ifc).name)
        for v in sensors:
            m = v["mapping"]
            print(f"{v['name']:22s} {v['quantity']:12s} model {v['position']}  "
                  f"{m['member']:11s} depth {m['depth_behind_cavity_face_m']:.3f} m  station {m['station_m']} m")
        print(f"written {out}")


if __name__ == "__main__":
    main()
