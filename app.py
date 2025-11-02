from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Structural API – PyNite style")

class Beam(BaseModel):
    L_m: float
    w_kN_m: float
    E_GPa: float = 210.0
    I_m4: float = 8e-4
    A_m2: float = 0.02
    limit_L_over: float = 300.0

class BeamBatch(BaseModel):
    beams: List[Beam]

@app.post("/beam/check")
def beam_check(batch: BeamBatch):
    results = []
    for b in batch.beams:
        L = b.L_m
        w = b.w_kN_m * 1e3
        E = b.E_GPa * 1e9
        I = b.I_m4
        limit = L / b.limit_L_over
        # formule base trave appoggiata con carico uniforme
        Mmax = w * L**2 / 8
        delta = 5 * w * L**4 / (384 * E * I)
        ok = delta <= limit
        results.append({
            "L_m": L,
            "w_kN_m": b.w_kN_m,
            "E_GPa": b.E_GPa,
            "I_m4": b.I_m4,
            "A_m2": b.A_m2,
            "Mmax_kNm": Mmax / 1e3,
            "delta_max_mm": delta * 1000,
            "limit_mm": limit * 1000,
            "check_ok": ok
        })
    summary = {
        "n": len(results),
        "ok": sum(1 for r in results if r["check_ok"]),
        "ko": sum(1 for r in results if not r["check_ok"]),
        "mean_delta_mm": sum(r["delta_max_mm"] for r in results) / len(results)
    }
    return {"results": results, "summary": summary}
