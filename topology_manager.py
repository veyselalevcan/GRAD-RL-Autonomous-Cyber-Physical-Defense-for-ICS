"""
topology_manager.py — GRAD-RL Framework (v6.0)
Graph-Based Asset & Vulnerability Mapper

Changes vs v5.0:
  - WADI dataset support (parallel topology, Zone 1/2/3)
  - Regex-based WADI tag parser
  - Dataset-agnostic map_asset_to_stage() dispatcher
  - CONFIG dict: switch datasets by changing one flag
  - Physical consequences & attack_mapping extended for WADI stages
"""

import networkx as nx
import json
import re
import os

# ─────────────────────────────────────────────
#  GLOBAL CONFIGURATION FLAG
#  Set DATASET = "SWaT" or "WADI" to switch context.
#  All downstream logic branches on this value.
# ─────────────────────────────────────────────
CONFIG = {
    "DATASET": "SWaT",   # Options: "SWaT" | "WADI"
    "TOPOLOGY_FILE_SWaT": "topology_config.json",
    "TOPOLOGY_FILE_WADI": "topology_wadi_config.json",
}

# ─────────────────────────────────────────────
#  WADI Tag Regex Pattern
#  Format: {zone}_{type}_{number}_{suffix}
#  Examples: 1_AIT_001_PV, 2_MV_003_STATUS, 3_P_001_STATUS
#  Zone    : 1 (Primary), 2 (Consumer), 3 (Return)
#  Types   : AIT, FIT, LIT, MV, P, PIT, TT, LS, HS, DPIT
#  Suffixes: PV, STATUS, SP, CV, MAN, AL, AH
# ─────────────────────────────────────────────
_WADI_TAG_RE = re.compile(
    r"^(?P<zone>[123])_"
    r"(?P<itype>[A-Z]+)_"
    r"(?P<num>\d{3})"
    r"(?:_\d+)?"             # optional sub-index (e.g. 1_P_001_1_STATUS)
    r"_(?P<suffix>[A-Z]+)$",
    re.IGNORECASE,
)


class GenericAssetMapper:
    """
    PhD Core Module: Graph-Based Asset & Vulnerability Mapper.

    Supports SWaT (sequential 6-stage pipeline) and
    WADI (parallel 3-zone distribution network).

    The dataset in use is controlled by CONFIG["DATASET"].
    """

    def __init__(self, dataset: str | None = None):
        """
        Args:
            dataset: Override CONFIG["DATASET"] at instantiation.
                     Useful for running both datasets in the same session.
        """
        self.dataset = dataset or CONFIG["DATASET"]
        self.graph = nx.DiGraph()

        self.vulnerability_db   = self._load_cvss_v4_definitions()
        self.knowledge_base     = self._load_xai_knowledge_base()

        # Stage-specific physical consequence maps (both datasets)
        self.physical_consequences_swat = {
            "Stage_P1": "Raw Water Tank (Tank-101) overflow causing flooding",
            "Stage_P2": "Incorrect chemical dosing (pH spike), creating toxic water",
            "Stage_P3": "Ultrafiltration (UF) pump cavitation and membrane rupture",
            "Stage_P4": "Dechlorination failure, causing permanent damage to RO units",
            "Stage_P5": "Reverse Osmosis (RO) high pressure buildup and pipe burst",
            "Stage_P6": "Contaminated water release to public distribution",
        }

        self.physical_consequences_wadi = {
            "SUPPLY_HEADER":        "Header tank (T-1XX) over/underflow causing supply-side pressure collapse",
            "SUPPLY_PUMP":          "Pump cavitation due to uncontrolled RPM change; mechanical seal failure",
            "DISTRIBUTION_MAIN":    "Loss of main distribution pressure; entire consumer network starved",
            "CONSUMER_TANK":        "Consumer-side tank overflow causing localised flooding",
            "DISTRIBUTION_VALVE":   "Valve stuck open/closed; uncontrolled or zero flow to consumer zone",
            "QUALITY_SENSOR":       "Contamination passing undetected to end consumers (health hazard)",
            "RETURN_PUMP":          "Return pump failure causing backpressure surge; pipe fatigue",
            "PRESSURE_ZONE":        "Pressure exceedance beyond design rating; pipe burst risk",
            "REMOTE_IO":            "Loss of remote monitoring visibility; operator blind to zone state",
            "TELEMETRY":            "Corrupted SCADA telemetry; automated responses based on false data",
        }

        # Unified translation layer (attack label → canonical attack type)
        self.attack_mapping = {
            # SWaT labels
            "Attack1": "Spoofing",
            "Attack2": "Injection",
            "Attack3": "PumpManipulation",
            "Attack4": "IntegrityViolation",
            "Attack5": "DoS",
            "Attack6": "SafetyBypass",
            "Attack30": "MultiVector",
            # WADI labels (iTrust naming convention)
            "Attack_P1": "Spoofing",
            "Attack_P2": "Injection",
            "Attack_1":  "Spoofing",
            "Attack_2":  "Injection",
            # Generic
            "MITM": "MITM", "Replay": "Replay",
            "Ransomware": "Ransomware", "Normal": "Normal",
        }

        # Load topology graph
        topo_file = (CONFIG["TOPOLOGY_FILE_WADI"]
                     if self.dataset == "WADI"
                     else CONFIG["TOPOLOGY_FILE_SWaT"])
        self._load_topology_from_json(topo_file)

        # If no external JSON loaded, build a default graph
        if len(self.graph.nodes) == 0:
            if self.dataset == "WADI":
                self._build_wadi_default_topology()
            else:
                self._build_swat_default_topology()

        self.centrality = nx.betweenness_centrality(self.graph)
        print(f"✅ TopologyManager [dataset={self.dataset}] loaded. "
              f"Nodes={self.graph.number_of_nodes()}, "
              f"Edges={self.graph.number_of_edges()}")

    # ──────────────────────────────────────────
    #  TOPOLOGY BUILDERS
    # ──────────────────────────────────────────

    def _load_topology_from_json(self, filename: str):
        config_path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            for node in data["nodes"]:
                self.graph.add_node(
                    node["id"],
                    type=node["type"],
                    criticality=node["criticality"],
                )
            for edge in data["edges"]:
                self.graph.add_edge(edge["source"], edge["target"])
        except Exception as e:
            print(f"⚠️  Topology JSON load failed ({filename}): {e}")

    def _build_swat_default_topology(self):
        """Sequential 6-stage SWaT pipeline."""
        stages = [f"Stage_P{i}" for i in range(1, 7)]
        plcs   = [f"PLC_P{i}"   for i in range(1, 7)]
        nodes  = stages + plcs + ["HMI_Main", "Historian"]

        crit = {**{s: 0.7 + 0.05 * i for i, s in enumerate(stages)},
                **{p: 0.9            for p in plcs},
                "HMI_Main": 1.0, "Historian": 0.6}

        for n in nodes:
            self.graph.add_node(n, type="ICS_Asset", criticality=crit.get(n, 0.5))

        # Sequential: Stage → PLC → next Stage
        for i in range(len(stages)):
            self.graph.add_edge(stages[i], plcs[i])
            self.graph.add_edge(plcs[i], "HMI_Main")
            if i < len(stages) - 1:
                self.graph.add_edge(plcs[i], stages[i + 1])
        self.graph.add_edge("HMI_Main", "Historian")

    def _build_wadi_default_topology(self):
        """
        Parallel WADI topology.

        Zone 1 (Primary Supply) feeds Distribution_Main.
        Distribution_Main fans out to two consumer branches (parallel).
        Zone 3 (Return) loops back to Supply for mass-balance.
        Quality_Control cross-connects to all zones (monitoring plane).
        """
        nodes_def = {
            # Zone 1 — Primary Supply
            "SUPPLY_HEADER":        ("Zone1_Supply",    1.0),
            "SUPPLY_PUMP":          ("Zone1_Supply",    0.9),
            "QUALITY_SENSOR":       ("Monitoring",      0.85),
            # Distribution
            "DISTRIBUTION_MAIN":    ("Distribution",    1.0),
            "DISTRIBUTION_VALVE":   ("Distribution",    0.8),
            # Zone 2 — Consumer (two parallel branches)
            "CONSUMER_TANK_A":      ("Zone2_Consumer",  0.75),
            "CONSUMER_TANK_B":      ("Zone2_Consumer",  0.75),
            "PRESSURE_ZONE":        ("Zone2_Consumer",  0.7),
            # Zone 3 — Return
            "RETURN_PUMP":          ("Zone3_Return",    0.65),
            "TELEMETRY":            ("Supervisory",     0.9),
            "REMOTE_IO":            ("Supervisory",     0.8),
            "HMI_Main":             ("Supervisory",     1.0),
        }
        for nid, (ntype, crit) in nodes_def.items():
            self.graph.add_node(nid, type=ntype, criticality=crit)

        edges = [
            # Supply → Distribution (critical bridge)
            ("SUPPLY_HEADER",     "SUPPLY_PUMP"),
            ("SUPPLY_PUMP",       "DISTRIBUTION_MAIN"),
            # Quality cross-connects (monitoring plane)
            ("QUALITY_SENSOR",    "DISTRIBUTION_MAIN"),
            ("QUALITY_SENSOR",    "HMI_Main"),
            # Distribution → consumer branches (PARALLEL)
            ("DISTRIBUTION_MAIN", "DISTRIBUTION_VALVE"),
            ("DISTRIBUTION_VALVE","CONSUMER_TANK_A"),
            ("DISTRIBUTION_VALVE","CONSUMER_TANK_B"),
            ("DISTRIBUTION_MAIN", "PRESSURE_ZONE"),
            # Return loop
            ("CONSUMER_TANK_A",   "RETURN_PUMP"),
            ("CONSUMER_TANK_B",   "RETURN_PUMP"),
            ("RETURN_PUMP",       "SUPPLY_HEADER"),   # mass-balance loop
            # Supervisory plane
            ("DISTRIBUTION_MAIN", "TELEMETRY"),
            ("SUPPLY_HEADER",     "TELEMETRY"),
            ("TELEMETRY",         "REMOTE_IO"),
            ("REMOTE_IO",         "HMI_Main"),
        ]
        self.graph.add_edges_from(edges)

    # ──────────────────────────────────────────
    #  TAG PARSERS  (dataset-agnostic dispatcher)
    # ──────────────────────────────────────────

    def map_asset_to_stage(self, asset_tag: str) -> str:
        """Dataset-agnostic dispatcher.  Routes to SWaT or WADI parser."""
        if self.dataset == "WADI":
            return self._map_wadi_tag(asset_tag)
        return self._map_swat_tag(asset_tag)

    def _map_swat_tag(self, swat_tag: str) -> str:
        """Original SWaT tag → Stage_P{n} mapping."""
        tag = str(swat_tag).upper()
        for i in range(1, 7):
            s = str(i)
            if any(f"{inst}{s}" in tag for inst in ["FIT", "MV", "LIT", "P"]):
                return f"Stage_P{i}"
        return "Stage_P1"

    def _map_wadi_tag(self, wadi_tag: str) -> str:
        """
        Regex-based WADI tag → functional stage mapping.

        Instrument type × zone → stage:
        ┌─────────────┬───────────────────────────────────────────────────┐
        │ Zone 1      │ LIT/FIT → SUPPLY_HEADER                          │
        │             │ P/MV    → SUPPLY_PUMP                            │
        │             │ AIT     → QUALITY_SENSOR                         │
        │             │ PIT     → PRESSURE_ZONE                          │
        ├─────────────┼───────────────────────────────────────────────────┤
        │ Zone 2      │ LIT/FIT → CONSUMER_TANK_A/B (alternating by num) │
        │             │ MV      → DISTRIBUTION_VALVE                      │
        │             │ P       → DISTRIBUTION_MAIN                       │
        │             │ AIT     → QUALITY_SENSOR                          │
        │             │ PIT     → PRESSURE_ZONE                           │
        ├─────────────┼───────────────────────────────────────────────────┤
        │ Zone 3      │ P       → RETURN_PUMP                             │
        │             │ FIT/LIT → RETURN_PUMP                             │
        │             │ Others  → TELEMETRY                               │
        └─────────────┴───────────────────────────────────────────────────┘
        """
        tag = str(wadi_tag).strip().upper()
        m = _WADI_TAG_RE.match(tag)

        if not m:
            # Fallback: try substring heuristics
            return self._wadi_fallback_map(tag)

        zone  = int(m.group("zone"))
        itype = m.group("itype")
        num   = int(m.group("num"))

        if zone == 1:
            if itype in ("LIT", "FIT"):   return "SUPPLY_HEADER"
            if itype in ("P", "MV"):      return "SUPPLY_PUMP"
            if itype == "AIT":            return "QUALITY_SENSOR"
            if itype == "PIT":            return "PRESSURE_ZONE"
            return "SUPPLY_HEADER"

        if zone == 2:
            if itype == "AIT":            return "QUALITY_SENSOR"
            if itype == "PIT":            return "PRESSURE_ZONE"
            if itype == "MV":             return "DISTRIBUTION_VALVE"
            if itype == "P":              return "DISTRIBUTION_MAIN"
            if itype in ("LIT", "FIT"):
                # Alternate between two consumer tanks by tag number parity
                return "CONSUMER_TANK_A" if num % 2 == 1 else "CONSUMER_TANK_B"
            return "DISTRIBUTION_MAIN"

        # zone == 3 (return network)
        if itype in ("P", "FIT", "LIT"): return "RETURN_PUMP"
        return "TELEMETRY"

    def _wadi_fallback_map(self, tag: str) -> str:
        """Substring-based fallback for non-standard WADI tag formats."""
        if "AIT" in tag:   return "QUALITY_SENSOR"
        if "PIT" in tag:   return "PRESSURE_ZONE"
        if "_3_" in tag:   return "RETURN_PUMP"
        if "MV"  in tag:   return "DISTRIBUTION_VALVE"
        if "LIT" in tag or "FIT" in tag:
            return "SUPPLY_HEADER" if "_1_" in tag else "CONSUMER_TANK_A"
        return "DISTRIBUTION_MAIN"

    # Convenience alias kept for backward compatibility with SWaT callers
    def map_swat_to_generic(self, swat_tag: str) -> str:
        return self._map_swat_tag(swat_tag)

    # ──────────────────────────────────────────
    #  GRAPH ANALYSIS
    # ──────────────────────────────────────────

    def trace_propagation_path(self, start_node: str) -> list[str]:
        try:
            return nx.shortest_path(self.graph, source=start_node, target="HMI_Main")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [start_node, "HMI_Main"]

    def get_critical_bridges(self) -> list[str]:
        """Return nodes with above-median betweenness centrality (structural chokepoints)."""
        median = sorted(self.centrality.values())[len(self.centrality) // 2]
        return [n for n, c in self.centrality.items() if c > median]

    # ──────────────────────────────────────────
    #  IMPACT ASSESSMENT (dataset-agnostic)
    # ──────────────────────────────────────────

    def assess_impact(self, asset_tag: str, raw_attack_type: str) -> dict:
        """
        Returns risk assessment dict.  Works for both SWaT and WADI asset tags.
        The dataset is determined by self.dataset (set at construction).
        """
        target_node       = self.map_asset_to_stage(asset_tag)
        general_attack    = self.attack_mapping.get(raw_attack_type, raw_attack_type)

        kb_entry   = self.knowledge_base.get(general_attack,  self.knowledge_base["Unknown"])
        cvss_data  = self.vulnerability_db.get(general_attack, self.vulnerability_db["Unknown"])

        # Select physical consequence map
        phys_map   = (self.physical_consequences_wadi
                      if self.dataset == "WADI"
                      else self.physical_consequences_swat)
        consequence = phys_map.get(target_node, "critical process instability")

        # Controller/PLC node name depends on dataset topology
        if self.dataset == "WADI":
            controller_node = "HMI_Main"  # WADI is SCADA-managed, not PLC-per-stage
        else:
            controller_node = target_node.replace("Stage", "PLC")

        story = (
            f"⚠️ **ALERT [{self.dataset}]:** A **{cvss_data['name']}** "
            f"({general_attack}) detected on **{asset_tag}** "
            f"→ mapped to **{target_node}**.\n"
            f"🔍 **DIAGNOSIS:** {controller_node} will receive corrupted process values. "
            f"Signature matches **{kb_entry['cwe']}**.\n\n"
            f"🔥 **FORECAST:** Unmitigated propagation → **{consequence}**.\n\n"
            f"🛡️ **IMMEDIATE ACTION:** {kb_entry['mitigation'][0]}."
        )

        node_cent  = self.centrality.get(target_node, 0.0)
        impacts    = cvss_data["impact_vectors"]
        base_impact = (impacts["A"] + impacts["I"] + impacts["S"] * 1.5) / 3.5
        risk_score  = min(base_impact * (1.0 + node_cent) * 10, 10.0)

        return {
            "dataset":           self.dataset,
            "risk_score":        round(risk_score, 2),
            "explanation":       story,
            "target_generic":    target_node,
            "attack_type":       general_attack,
            "betweenness":       round(node_cent, 4),
            "mitigation":        kb_entry["mitigation"],
            "propagation_path":  self.trace_propagation_path(target_node),
            "is_bridge_node":    target_node in self.get_critical_bridges(),
        }

    # ──────────────────────────────────────────
    #  KNOWLEDGE BASES (shared across datasets)
    # ──────────────────────────────────────────

    def _load_xai_knowledge_base(self) -> dict:
        return {
            "Spoofing":          {"cwe": "CWE-20",   "mitigation": ["Switch to MANUAL mode", "Cross-validate redundant sensors"]},
            "Injection":         {"cwe": "CWE-77",   "mitigation": ["Isolate PLC from Network", "Revoke compromised credentials"]},
            "PumpManipulation":  {"cwe": "CWE-1299", "mitigation": ["Check Pump RPM Logs", "Emergency E-Stop if vibration detected"]},
            "IntegrityViolation":{"cwe": "CWE-254",  "mitigation": ["Restore Last Known Good Config", "Enable Integrity Checks (CRC)"]},
            "DoS":               {"cwe": "CWE-400",  "mitigation": ["Enable Rate Limiting", "Switch to Local Control Panel"]},
            "SafetyBypass":      {"cwe": "CWE-805",  "mitigation": ["IMMEDIATE SYSTEM SHUTDOWN", "Engage Mechanical Safety Locks"]},
            "MITM":              {"cwe": "CWE-300",  "mitigation": ["Rotate Encryption Keys", "Check ARP Table"]},
            "Replay":            {"cwe": "CWE-294",  "mitigation": ["Implement Timestamp Checks", "Reset Session Keys"]},
            "Ransomware":        {"cwe": "CWE-1269", "mitigation": ["Sever Network Connections", "Restore from Offline Backups"]},
            "MultiVector":       {"cwe": "CWE-Multiple", "mitigation": ["Activate Incident Response Plan", "Isolate All External Connections"]},
            "Unknown":           {"cwe": "CWE-ZeroDay", "mitigation": ["Manual Inspection"]},
            "Normal":            {"cwe": "N/A",      "mitigation": []},
        }

    def _load_cvss_v4_definitions(self) -> dict:
        # CVSS v4.0 impact vectors: S=Safety, A=Availability, I=Integrity
        # IEC 61511 safety multiplier applied to S (1.5×) during risk_score calculation
        return {
            "Spoofing":          {"name": "Sensor Spoofing",       "impact_vectors": {"S": 0.8, "A": 0.4, "I": 0.9}},
            "Injection":         {"name": "Actuator Injection",    "impact_vectors": {"S": 0.9, "A": 0.6, "I": 0.9}},
            "PumpManipulation":  {"name": "Pump Manipulation",     "impact_vectors": {"S": 0.8, "A": 0.9, "I": 0.7}},
            "IntegrityViolation":{"name": "Data Integrity Loss",   "impact_vectors": {"S": 0.7, "A": 0.7, "I": 1.0}},
            "DoS":               {"name": "Denial of Service",     "impact_vectors": {"S": 0.6, "A": 0.9, "I": 0.0}},
            "SafetyBypass":      {"name": "Safety Logic Bypass",   "impact_vectors": {"S": 1.0, "A": 1.0, "I": 1.0}},
            "MITM":              {"name": "Man-in-the-Middle",     "impact_vectors": {"S": 0.7, "A": 0.7, "I": 0.9}},
            "Replay":            {"name": "Replay Attack",         "impact_vectors": {"S": 0.6, "A": 0.5, "I": 0.8}},
            "Ransomware":        {"name": "Ransomware",            "impact_vectors": {"S": 0.5, "A": 1.0, "I": 1.0}},
            "Unknown":           {"name": "Anomaly",               "impact_vectors": {"S": 0.0, "A": 0.0, "I": 0.0}},
            "Normal":            {"name": "Normal",                "impact_vectors": {"S": 0.0, "A": 0.0, "I": 0.0}},
        }