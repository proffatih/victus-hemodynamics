"""Fill numeric placeholders in manuscript.tex from the result CSVs."""
import os, csv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
TEX = os.path.join(HERE, '..', 'manuscript', 'manuscript.tex')


def rd(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.reader(f))


stiff = rd('study_stiffness.csv')[1:]
sten = rd('study_stenosis.csv')[1:]

# stiffness table rows: E_Pa,c0,PWV,SBP,DBP,PP,AIx
stiff_rows = []
for r in stiff:
    E_kPa = float(r[0]) / 1000
    stiff_rows.append(
        f"{E_kPa:.0f} & {float(r[1]):.2f} & {float(r[2]):.2f} & "
        f"{float(r[3]):.0f}/{float(r[4]):.0f} & {float(r[5]):.1f} & {float(r[6]):.1f}\\\\")
stiff_table = "\n".join(stiff_rows)

# stenosis table rows: area_red,diam_red,dP,Qd,damp
sten_rows = []
for r in sten:
    sten_rows.append(
        f"{float(r[0]):.0f} & {float(r[1]):.1f} & {float(r[2]):.2f} & "
        f"{float(r[3]):.1f} & {float(r[4]):.1f}\\\\")
sten_table = "\n".join(sten_rows)

# headline numbers
pwv_low, pwv_high = float(stiff[0][2]), float(stiff[-1][2])
aix_low, aix_high = float(stiff[0][6]), float(stiff[-1][6])
pp_low, pp_high = float(stiff[0][5]), float(stiff[-1][5])
dp_high = float(sten[-1][2])
damp_high = float(sten[-1][4])

with open(TEX) as f:
    s = f.read()

repl = {
    'PLACEHOLDER\\_STIFF\\_TABLE': stiff_table,
    'PLACEHOLDER\\_STEN\\_TABLE': sten_table,
    'PLACEHOLDER\\_AIX\\_LOW': f'{aix_low:.1f}',
    'PLACEHOLDER\\_AIX\\_HIGH': f'{aix_high:.1f}',
    'PLACEHOLDER\\_PWV\\_LOW': f'{pwv_low:.1f}',
    'PLACEHOLDER\\_PWV\\_HIGH': f'{pwv_high:.1f}',
    'PLACEHOLDER\\_PP\\_LOW': f'{pp_low:.0f}',
    'PLACEHOLDER\\_PP\\_HIGH': f'{pp_high:.0f}',
    'PLACEHOLDER\\_DP\\_HIGH': f'{dp_high:.1f}',
    'PLACEHOLDER\\_DAMP\\_HIGH': f'{damp_high:.0f}',
}
for k, v in repl.items():
    s = s.replace(k, v)

with open(TEX, 'w') as f:
    f.write(s)
print("Filled placeholders.")
print(f"  AIx {aix_low:.1f}% -> {aix_high:.1f}% over PWV {pwv_low:.1f}->{pwv_high:.1f} m/s")
print(f"  PP {pp_low:.0f} -> {pp_high:.0f} mmHg")
print(f"  stenosis dP_high {dp_high:.1f} mmHg, damping {damp_high:.0f}%")
remaining = s.count('PLACEHOLDER')
print(f"  remaining PLACEHOLDER tokens: {remaining}")
