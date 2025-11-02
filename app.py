from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse
from PyNiteFEA import FEModel3D
import io
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI(title="Structural API – PyNiteFEA", version="1.0.0")

class Beam(BaseModel):
    L_m: float
    w_kN_m: float
    E_GPa: float = 210.0
    I_m4: float = 8e-4
    A_m2: float = 0.02
    limit_L_over: float = 300.0
    samples: int = 101

class BeamBatch(BaseModel):
    beams: List[Beam]


@app.post("/beam/check")
def beam_check(batch: BeamBatch):
    """Analisi strutturale FEM con PyNiteFEA per una trave appoggiata con carico distribuito"""
    results = []

    for b in batch.beams:
        L = float(b.L_m)
        w = float(b.w_kN_m) * 1e3     # N/m
        E = float(b.E_GPa) * 1e9
        I = float(b.I_m4)
        A = float(b.A_m2)
        limit = L / float(b.limit_L_over)

        # Modello PyNite
        model = FEModel3D()
        model.AddNode("i", 0, 0, 0)
        model.AddNode("j", L, 0, 0)
        model.AddMaterial("steel", E, E/2.6, 7850)
        model.AddMember("beam", "i", "j", "steel", A, I, I*0.6, I*0.1)
        model.DefineSupport("i", True, True, True, False, False, False)
        model.DefineSupport("j", True, True, True, False, False, False)
        model.AddMemberDistLoad("beam", Direction="Fy", w1=-w, w2=-w, x1=0, x2=L)
        model.Analyze(check_statics=True)

        member = model.GetMember("beam")
        Mmax = max(abs(member.max_moment("My")), abs(member.min_moment("My")))
        Vmax = max(abs(member.max_shear("Fy")), abs(member.min_shear("Fy")))
        delta_max = abs(member.max_deflection("dFy"))
        limit_mm = limit * 1000
        ok = delta_max <= limit

        out = {
            "L_m": L,
            "w_kN_m": b.w_kN_m,
            "E_GPa": b.E_GPa,
            "I_m4": b.I_m4,
            "A_m2": b.A_m2,
            "Mmax_kNm": Mmax / 1e3,
            "Vmax_kN": Vmax / 1e3,
            "delta_max_mm": delta_max * 1000,
            "limit_mm": limit_mm,
            "check_ok": ok
        }
        results.append(out)

    summary = {
        "n": len(results),
        "ok": sum(1 for r in results if r["check_ok"]),
        "ko": sum(1 for r in results if not r["check_ok"]),
        "mean_delta_mm": sum(r["delta_max_mm"] for r in results) / len(results)
    }

    return {"results": results, "summary": summary}


@app.post("/beam/plot")
def beam_plot(beam: Beam):
    """Genera il grafico FEM (momento, taglio, deformata)"""
    L = float(beam.L_m)
    w = float(beam.w_kN_m) * 1e3
    E = float(beam.E_GPa) * 1e9
    I = float(beam.I_m4)
    A = float(beam.A_m2)

    model = FEModel3D()
    model.AddNode("i", 0, 0, 0)
    model.AddNode("j", L, 0, 0)
    model.AddMaterial("steel", E, E/2.6, 7850)
    model.AddMember("beam", "i", "j", "steel", A, I, I*0.6, I*0.1)
    model.DefineSupport("i", True, True, True, False, False, False)
    model.DefineSupport("j", True, True, True, False, False, False)
    model.AddMemberDistLoad("beam", Direction="Fy", w1=-w, w2=-w, x1=0, x2=L)
    model.Analyze(check_statics=True)

    member = model.GetMember("beam")
    x = np.linspace(0, L, beam.samples)
    V = [member.shear("Fy", xi) for xi in x]
    M = [member.moment("My", xi) for xi in x]
    y = [member.deflection("dFy", xi) for xi in x]

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    axs[0].plot(x, V)
    axs[0].set_title("Taglio V(x) [N]")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(x, M)
    axs[1].set_title("Momento M(x) [N·m]")
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(x, y)
    axs[2].set_title("Deformata δ(x) [m]")
    axs[2].grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
