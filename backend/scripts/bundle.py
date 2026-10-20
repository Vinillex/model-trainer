"""Portable snapshot of the database and the stored files, so a project can move to another machine through git.

    python -m scripts.bundle export                 # from backend/: reads the configured DB + local storage
    python -m scripts.bundle import                 # into an EMPTY database; ids are preserved

The bundle (repo-root `handover-data/` by default) is plain JSON plus the files, so it diffs sensibly in git, doesn't depend on
the SQLite file format or on the schema staying exactly the same (there are no migrations yet), and the files can be restored
into any storage backend. Every table is dumped, so tables added later travel too. Output only ever shows counts, never values.
"""

import argparse
import enum
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, Engine, func, select

from app.config import settings
from app.db import Base
from app import models  # noqa: F401 - registers every table on Base.metadata
from app.storage import LocalStorage, Storage, get_storage

FORMAT = 1
DB_FILE, FILES_FILE, STORAGE_DIR = "db.json", "files.json", "storage"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "handover-data"
JUNK_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}  # folder-view files Finder/Explorer drop into any folder you open


def _is_junk(path: Path) -> bool:
    return path.name in JUNK_FILES or path.name.startswith("._")  # "._x" = macOS AppleDouble sidecars


class BundleError(Exception):
    pass


def _dump_value(column, value):
    if value is None:
        return None
    if isinstance(value, enum.Enum):  # before str: the app's enums are also str subclasses
        return value.name  # Enum columns store member names
    if isinstance(column.type, DateTime):
        return value.isoformat()
    if isinstance(column.type, JSON) or isinstance(value, (str, int, float, bool)):
        return value
    raise BundleError(f"don't know how to export {type(value).__name__} in {column.table.name}.{column.name}")


def _load_value(column, value):
    if value is not None and isinstance(column.type, DateTime):
        return datetime.fromisoformat(value)
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_bundle(engine: Engine, storage_root: Path, out: Path) -> dict:
    """Write the database and every stored file under `storage_root` into `out`. Safe to repeat: it replaces its own output."""
    if out.exists() and any(out.iterdir()) and not (out / DB_FILE).exists():
        raise BundleError(f"{out} isn't empty and doesn't look like a bundle; refusing to write into it")
    for name in (DB_FILE, FILES_FILE):
        (out / name).unlink(missing_ok=True)
    _rmtree(out / STORAGE_DIR)
    out.mkdir(parents=True, exist_ok=True)

    tables, counts = {}, {}
    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            rows = conn.execute(select(table).order_by(*table.primary_key.columns)).all()  # primary-key order: stable diffs
            tables[table.name] = [{c.name: _dump_value(c, row._mapping[c]) for c in table.columns} for row in rows]
            counts[table.name] = len(rows)

    files = {}
    for path in sorted(p for p in storage_root.rglob("*") if p.is_file() and not _is_junk(p)):
        key = path.relative_to(storage_root).as_posix()
        data = path.read_bytes()
        dest = out / STORAGE_DIR / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files[key] = {"sha256": _sha256(data), "size": len(data)}

    _write_json(out / DB_FILE, {"format": FORMAT, "tables": tables})
    _write_json(out / FILES_FILE, files)
    return {"tables": counts, "files": len(files), "bytes": sum(f["size"] for f in files.values())}


def import_bundle(engine: Engine, storage: Storage, bundle: Path) -> dict:
    """Restore a bundle into an empty SQLite database and into `storage`, keeping every id."""
    if engine.dialect.name != "sqlite":
        raise BundleError("import supports SQLite only (other databases would need their id sequences reset afterwards)")
    db = json.loads((bundle / DB_FILE).read_text(encoding="utf-8")) if (bundle / DB_FILE).exists() else None
    if db is None or db.get("format") != FORMAT:
        raise BundleError(f"{bundle} is not a format-{FORMAT} bundle (no {DB_FILE})")
    files = json.loads((bundle / FILES_FILE).read_text(encoding="utf-8"))

    # Check every file before touching anything: a git checkout that rewrote line endings or lost an LFS file shows up here.
    blobs = {}
    for key, meta in files.items():
        path = bundle / STORAGE_DIR / key
        if not path.exists():
            raise BundleError(f"bundle file missing: {key}")
        data = path.read_bytes()
        if _sha256(data) != meta["sha256"]:
            raise BundleError(f"bundle file is corrupted (checksum differs): {key}")
        blobs[key] = data

    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        used = [t.name for t in Base.metadata.sorted_tables if conn.execute(select(func.count()).select_from(t)).scalar()]
    if used:
        raise BundleError(f"the database isn't empty ({', '.join(used)} already have rows); import into a fresh database")

    skipped_tables = sorted(set(db["tables"]) - {t.name for t in Base.metadata.sorted_tables})
    dropped_columns = 0
    for key, data in blobs.items():  # before the rows: re-running after a failure just overwrites
        storage.put(key, data)

    counts = {}
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            rows = db["tables"].get(table.name, [])
            known = {c.name: c for c in table.columns}
            prepared = []
            for row in rows:
                dropped_columns += sum(1 for name in row if name not in known)
                prepared.append({name: _load_value(known[name], value) for name, value in row.items() if name in known})
            if prepared:
                conn.execute(table.insert(), prepared)
            counts[table.name] = len(prepared)
    return {"tables": counts, "files": len(blobs), "skipped_tables": skipped_tables, "dropped_columns": dropped_columns}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        p.rmdir() if p.is_dir() else p.unlink()
    path.rmdir()


def _summary(result: dict) -> str:
    rows = ", ".join(f"{t} {n}" for t, n in result["tables"].items() if n)
    return f"{rows or 'no rows'}; {result['files']} files"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="write the bundle").add_argument("--dir", type=Path, default=DEFAULT_DIR)
    sub.add_parser("import", help="restore the bundle into an empty database").add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "export":
            if settings.storage_backend != "local":
                raise BundleError("export reads local storage only (MT_STORAGE_BACKEND=local)")
            from app.db import engine
            r = export_bundle(engine, LocalStorage(settings.local_storage_dir).root, args.dir)
            print(f"exported to {args.dir}: {_summary(r)} ({r['bytes'] / 2**20:.1f} MB)")
        else:
            from app.db import engine
            Path("data").mkdir(exist_ok=True)
            r = import_bundle(engine, get_storage(), args.dir)
            print(f"imported from {args.dir}: {_summary(r)}")
            if r["skipped_tables"] or r["dropped_columns"]:
                print(f"note: the bundle is from a different schema; skipped tables {r['skipped_tables']}, dropped {r['dropped_columns']} column values")
    except BundleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
