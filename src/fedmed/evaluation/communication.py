"""Measured communication payloads.

Two complementary measurements, both of real serialised bytes:

* ``parameters_nbytes``: size of the serialised model tensors in a Flower
  ``Parameters`` object (what FedAvg conceptually sends/receives).
* ``make_comm_meter_mod``: a Flower client mod that records, for every message a
  simulated client receives and sends, the size of its Protocol Buffers
  encoding (``RecordDict`` -> proto ``ByteSize``). This includes SecAgg+ key
  exchange messages and metric/config records.

Neither figure includes gRPC/HTTP-2 framing, TLS, or TCP/IP overhead, and in
simulation no bytes cross a network, so no network transfer *time* is measured.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flwr.app import Context, Message, RecordDict
from flwr.common import Parameters
from flwr.common.serde import recorddict_to_proto

SECAGG_CONFIG_KEY = "secaggplus_configs"
SECAGG_STAGE_KEY = "stage"


def parameters_nbytes(parameters: Parameters) -> int:
    return int(sum(len(t) for t in parameters.tensors))


def recorddict_nbytes(content: RecordDict) -> int:
    return int(recorddict_to_proto(content).ByteSize())


def message_content_nbytes(msg: Message) -> int:
    return recorddict_nbytes(msg.content) if msg.has_content() else 0


def _stage_of(msg: Message) -> str:
    mtype = str(msg.metadata.message_type)
    if msg.has_content():
        cfg = msg.content.config_records.get(SECAGG_CONFIG_KEY)
        if cfg is not None and SECAGG_STAGE_KEY in cfg:
            return f"{mtype}:secagg_{cfg[SECAGG_STAGE_KEY]}"
    return mtype


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record) + "\n"
    # O_APPEND writes of a single short line are effectively atomic on POSIX.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def make_comm_meter_mod(log_path: str):
    """Build a client mod that logs serialised message sizes to a JSONL file."""
    path = Path(log_path)

    def comm_meter_mod(msg: Message, ctx: Context, call_next) -> Message:
        stage = _stage_of(msg)  # read before inner mods consume the stage key
        rec = {
            "client_id": int(ctx.node_config.get("partition-id", -1)),
            "round": msg.metadata.group_id,
            "stage": stage,
            "downlink_bytes": message_content_nbytes(msg),
            "t": time.time(),
        }
        try:
            out = call_next(msg, ctx)
        except Exception:
            rec.update(uplink_bytes=0, status="client_error")
            append_jsonl(path, rec)
            raise
        rec.update(uplink_bytes=message_content_nbytes(out), status="ok" if out.has_content() else "error")
        append_jsonl(path, rec)
        return out

    return comm_meter_mod
