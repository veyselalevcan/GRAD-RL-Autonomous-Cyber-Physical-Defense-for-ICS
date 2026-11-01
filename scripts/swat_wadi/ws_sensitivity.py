import numpy as np
import pandas as pd

# topology_manager.py içindeki CVSS v4.0 etki vektörleri
vulnerabilities = {
    "Spoofing":          {"S": 0.8, "A": 0.4, "I": 0.9},
    "Injection":         {"S": 0.9, "A": 0.6, "I": 0.9},
    "PumpManipulation":  {"S": 0.8, "A": 0.9, "I": 0.7},
    "IntegrityViolation":{"S": 0.7, "A": 0.7, "I": 1.0},
    "DoS":               {"S": 0.6, "A": 0.9, "I": 0.0},
    "SafetyBypass":      {"S": 1.0, "A": 1.0, "I": 1.0},
    "MITM":              {"S": 0.7, "A": 0.7, "I": 0.9},
}

# Simüle edilecek Düğüm Merkeziliği (Node Centrality) dağılımı (0.0 ile 0.3 arası 100 farklı düğüm)
centrality_distribution = np.linspace(0.0, 0.3, 100)
CRITICAL_THRESHOLD = 9.0

results = []
ws_values = [1.2, 1.5, 2.0]

for ws in ws_values:
    gate_activations = 0
    total_scenarios = 0
    
    # Sadece en kritik iki saldırı için ortalama CVSS skorunu da makale için saklayalım
    t0836_cvss = 0  # SafetyBypass için temsili
    t0856_cvss = 0  # Spoofing için temsili
    
    for vuln, impacts in vulnerabilities.items():
        # Formül: (A + I + S * ws) / (2.0 + ws) -> Max değerin 1.0 kalması için normalize ediyoruz
        base_impact = (impacts["A"] + impacts["I"] + impacts["S"] * ws) / (2.0 + ws)
        
        if vuln == "SafetyBypass": t0836_cvss = base_impact * 10
        if vuln == "Spoofing": t0856_cvss = base_impact * 10
            
        for cent in centrality_distribution:
            # topology_manager.py'deki risk formülü
            risk_score = min(base_impact * (1.0 + cent) * 10, 10.0)
            
            if risk_score >= CRITICAL_THRESHOLD:
                gate_activations += 1
            total_scenarios += 1
            
    results.append({
        "w_s Parameter": ws,
        "SafetyBypass (T0836) Risk": f"{t0836_cvss:.2f}",
        "Spoofing (T0856) Risk": f"{t0856_cvss:.2f}",
        "Gate Activations (Risk ≥ 9.0)": f"{gate_activations} / {total_scenarios}"
    })

# Çıktıyı makaleye eklenecek tablo formatında yazdır
df_results = pd.DataFrame(results)
print("\n" + "="*70)
print("  MAKALE İÇİN HASSASİYET ANALİZİ TABLOSU (SENSITIVITY ANALYSIS)")
print("="*70)
print(df_results.to_string(index=False))
print("\n[!] YORUM: Bu tabloyu hocanın geri bildirimindeki eksik tablo yerine koyabilirsin.")