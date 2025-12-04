from scapy.all import rdpcap, TCP, UDP, IP
import pandas as pd
import numpy as np
import time

def extract_flows_from_pcap(file_path):
    """
    Parse a pcap file and extract flow-based features compatible with ML model.
    """

    packets = rdpcap(file_path)

    flows = {}  # key = (src, dst, sport, dport, proto)

    for pkt in packets:
        if IP not in pkt:
            continue

        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto

        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

        key = (src, dst, sport, dport, proto)

        ts = float(pkt.time)
        size = len(pkt)

        if key not in flows:
            flows[key] = {
                "start": ts,
                "end": ts,
                "total_bytes": 0,
                "pkt_count": 0,
                "srv_count": 0,
                "src_bytes": 0,
                "dst_bytes": 0,
                "serror_count": 0,
                "rerror_count": 0
            }

        f = flows[key]
        f["end"] = ts
        f["pkt_count"] += 1
        f["total_bytes"] += size

        # Simple heuristic:
        if pkt[IP].src == src:
            f["src_bytes"] += size
        else:
            f["dst_bytes"] += size

        # Error estimates
        if TCP in pkt:
            if pkt[TCP].flags & 0x04:  # RST flag
                f["rerror_count"] += 1
            if pkt[TCP].flags & 0x02 and pkt[TCP].flags & 0x10 == 0:
                f["serror_count"] += 1

    rows = []

    for key, f in flows.items():
        duration = f["end"] - f["start"]

        row = {
            "duration": duration,
            "src_bytes": f["src_bytes"],
            "dst_bytes": f["dst_bytes"],
            "count": f["pkt_count"],
            "srv_count": f["srv_count"],  # not from pcap; kept 0 unless extended
            "serror_rate": f["serror_count"] / max(1, f["pkt_count"]),
            "srv_serror_rate": 0, 
            "rerror_rate": f["rerror_count"] / max(1, f["pkt_count"]),
            "srv_rerror_rate": 0,
            "same_srv_rate": 0.5,
            "diff_srv_rate": 0.1,
            "srv_diff_host_rate": 0.1,
            "dst_host_count": f["pkt_count"],
            "dst_host_srv_count": f["pkt_count"],
            "dst_host_same_srv_rate": 0.5,
            "dst_host_diff_srv_rate": 0.1
        }

        rows.append(row)

    # Convert to dataframe usable by ML models
    df = pd.DataFrame(rows)

    return df
