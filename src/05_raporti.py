"""Hapi 5: Gjeneron RAPORTI.md me rezultatet reale të projektit.

Nisja (pasi të kesh nisur hapat 1-4):
    python src/05_raporti.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.common import REPORTS, ROOT  # noqa: E402


EMRAT = dict(zip(["Jan", "Shk", "Mar", "Pri", "Maj", "Qer", "Kor", "Gus", "Sht", "Tet", "Nën", "Dhj"],
             ["janar", "shkurt", "mars", "prill", "maj", "qershor", "korrik", "gusht",
              "shtator", "tetor", "nëntor", "dhjetor"]))


def load(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main():
    p = load("pastrimi.json")
    g = load("grafiket.json")
    r = load("modelet_regresion.json")
    k = load("modelet_klasifikim.json")
    tab_r = (REPORTS / "tabela_regresion.md").read_text(encoding="utf-8")
    tab_k = (REPORTS / "tabela_klasifikim.md").read_text(encoding="utf-8")

    para_b, para_m = p["para"]["biciklete"], p["para"]["moti"]
    mungesa = p["pas"]["vlera_qe_mungojne"]
    mungesa_txt = (", ".join(f"`{c}`: {n}" for c, n in mungesa.items())
                   if mungesa else "asnjë")
    n = r["ndarja"]
    cm = k["matrica_konfuzionit"]["rreshtat=reale, kolonat=parashikuar"]
    rt, bt, kt = r["testim"], r["testim_modeli_baze"], k["testim"]
    kor = sorted(g["korrelacioni"].items(), key=lambda x: -abs(x[1]))[:3]
    kor_txt = ", ".join(f"`{c}` (r = {v})" for c, v in kor)
    perm = round(100 * (1 - rt["MAE"] / bt["MAE"]))
    perm_txt = f"{perm}% më pak gabim" if perm >= 0 else f"{-perm}% më shumë gabim"

    text = f"""# Raporti: UrbanBike AI

**Parashikimi i kërkesës ditore për biçikleta me qira me inteligjencë artificiale**

## 1. Qëllimi
Ndërtimi i një aplikacioni të thjeshtë që parashikon sa biçikleta do të merren me qira në një ditë dhe nëse kërkesa do të jetë **normale** apo **e lartë**, duke përdorur të dhëna reale historike për qiratë dhe motin.

## 2. Të dhënat
- **Qiratë:** UCI Bike Sharing Dataset (`day.csv`), Washington DC, {para_b['rreshta']} ditë.
- **Moti:** NASA POWER (`nasa_weather.csv`): temperatura, reshjet, era, {para_m['rreshta']} ditë.
- Periudha pas përgatitjes: **{p['periudha'][0]}** deri **{p['periudha'][1]}** ({p['pas']['rreshta']} ditë).

## 3. Kontrolli dhe pastrimi i të dhënave
| Kontrolli | Qiratë | Moti |
|---|---|---|
| Rreshta dublikatë | {para_b['rreshta_dublikate']} | {para_m['rreshta_dublikate']} |
| Data të përsëritura | {para_b['data_dublikate']} | {para_m['data_dublikate']} |
| Vlera që mungojnë | {sum(para_b['vlera_qe_mungojne'].values())} | {sum(para_m['vlera_qe_mungojne'].values())} |

- Datat u kthyen në tipin `datetime`, ndërsa kolonat kategorike (dita, muaji, ditë pune, festë) në numra të plotë.
- Vlerat `-999` të NASA-s u trajtuan si vlera që mungojnë dhe u plotësuan me medianën (`SimpleImputer`).
- Kolonat `casual` dhe `registered` u hoqën, sepse mbledhja e tyre jep vetë përgjigjen (rrjedhje informacioni).
- Variablat u ndërtuan **vetëm nga e kaluara**: qiratë e djeshme, qiratë e 7 ditëve më parë, mesatarja e 7 ditëve dhe moti i 7 ditëve më parë.
- Vlera që mungojnë pas përgatitjes: {mungesa_txt}.

## 4. Analiza me grafikë
- Kërkesa mesatare: **{g['mesatarja']:.0f}** qira në ditë. Muaji më i lartë: **{EMRAT[g['muaji_me_i_larte']]}**, më i ulëti: **{EMRAT[g['muaji_me_i_ulet']]}**.
- Variablat me lidhjen më të fortë me kërkesën: {kor_txt}.

![Kërkesa në kohë](reports/figures/01_kerkesa_ne_kohe.png)
![Sipas muajit](reports/figures/02_sipas_muajit.png)
![Moti](reports/figures/05_moti.png)

## 5. Modelet e regresionit
Ndarja sipas kohës, pa përzierje, që modeli të mos shohë të ardhmen:
- **Trajnim:** {n['trajnim'][0]} deri {n['trajnim'][1]}
- **Validim:** {n['validim'][0]} deri {n['validim'][1]}
- **Testim:** {n['testim'][0]} deri {n['testim'][1]}

Krahasimi i modeleve (modeli bazë **Yesterday** = qiratë e djeshme; **DummyMean** = mesatarja e trajnimit):

{tab_r}

Modeli i zgjedhur: **{r['modeli_i_zgjedhur']}**. Në **testim** ai arriti MAE = **{rt['MAE']}**, RMSE = {rt['RMSE']}, R² = {rt['R2']}, kundrejt MAE = {bt['MAE']} të modelit bazë Yesterday: **{perm_txt}**.

![Parashikimi në testim](reports/figures/08_parashikimi_testim.png)

## 6. Klasifikimi: kërkesë normale apo e lartë
Një ditë quhet **e lartë** kur kalon **{k['pragu_e_larte']:.0f}** qira (percentili 75 i trajnimit).

{tab_k}

Modeli i zgjedhur: **{k['modeli_i_zgjedhur']}**. Në testim: accuracy = **{kt['accuracy']}**, F1 = **{kt['F1']}**, precision = {kt['precision']}, recall = {kt['recall']}.

| | Parashikuar normale | Parashikuar e lartë |
|---|---|---|
| **Reale normale** | {cm[0][0]} | {cm[0][1]} |
| **Reale e lartë** | {cm[1][0]} | {cm[1][1]} |

## 7. Ndërfaqja dhe aplikacioni
- **Koncepti në Figma:** https://www.figma.com/design/rsgIsQKwTSy62LyyE0zue4
- **Aplikacioni lokal** (Streamlit) nisur me `streamlit run app.py`. Lejon zgjedhjen e një date historike dhe krahason parashikimin me vlerën reale, ose vendosjen e vlerave me dorë. Tregon numrin e parashikuar dhe etiketën normale / e lartë.

## 8. Kufizimet
- Vetëm **2 vite** të dhëna dhe një qytet (Washington DC). Modeli mund të mos përgjithësohet në qytete ose vite të tjera.
- Testimi mbulon vetëm vjeshtën dhe dimrin e 2012-s.
- Kërkesa rritet nga 2011 në 2012. Modelet me pemë (Random Forest) nuk ekstrapolojnë mirë një trend që vazhdon të rritet.
- Moti merret me vonesë 7-ditore. Parashikimi i motit për ditën e synuar mund të përmirësonte rezultatet.
- Pragu "e lartë" (percentili 75) është zgjedhje e thjeshtë dhe mund të përshtatet sipas nevojës së biznesit.

## 9. Kontributi im
<!-- SHKRUAJE VETË këtë pjesë me fjalët e tua. P.sh.: -->
- [Çfarë bëre vetë: pastrimi, grafikët, modelet, aplikacioni, Figma...]
- [Çfarë mësove: p.sh. pse ndarja sipas kohës dhe shmangia e rrjedhjes së informacionit janë të rëndësishme]
- [Çfarë vështirësish pate dhe si i zgjidhe]
- [Çfarë do të përmirësoje me më shumë kohë]

## 10. Si ta nisësh projektin
```
pip install -r requirements.txt
python src/01_pastrimi.py
python src/02_grafiket.py
python src/03_modelet.py
python src/04_klasifikimi.py
python src/05_raporti.py
streamlit run app.py
```
"""
    out = ROOT / "RAPORTI.md"
    if out.exists() and out.read_text(encoding="utf-8").strip():
        backup = ROOT / "RAPORTI_i_vjeter.md"
        if not backup.exists():
            shutil.copy(out, backup)
            print("RAPORTI.md i mëparshëm u ruajt si RAPORTI_i_vjeter.md")
    out.write_text(text, encoding="utf-8")
    print("U krijua:", out)
    print("Mos harro: shkruaj vetë seksionin 'Kontributi im'.")


if __name__ == "__main__":
    main()