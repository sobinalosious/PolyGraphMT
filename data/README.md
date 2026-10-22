# Data

`data/raw/` contains the CSV inputs used by the training scripts.

All `*_MD.csv` and `*_DFT.csv` files were generated with the ADEPT workflow
(all-atom MD in LAMMPS with GAFF2, and monomer DFT in Psi4), as described in
Alosious et al., *Digital Discovery* 2026, 5, 3021–3043. `CP_GC.csv` contains
group-contribution estimates.

Each CSV has two columns: `SMILES` (the polymer repeat unit, with `*` marking
the two connection points) and one value column. Some files contain a few
duplicate SMILES; the loader averages them.

**Values are stored exactly in the units listed under "Stored units" below.**
Three properties (`TD`, `DIF`, `VISC`) are stored with a scale factor, so read
that column before using the raw numbers.

## Property reference

| File | Column | Property | Symbol | Stored units | Value in physical units | n (unique SMILES) | Method and conditions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TD_MD.csv` | `TD` | Thermal diffusivity (**not** a temperature) | α<sub>T</sub> | 10⁻⁷ m² s⁻¹ | `TD × 1e-7` m² s⁻¹ | 799 | α<sub>T</sub> = κ / (ρ·C<sub>p</sub>), computed from ADEPT MD κ (`TC_MD`), ρ (`RHO_MD`) and MD C<sub>p</sub> at 300 K and 1 atm. The MD C<sub>p</sub> values are not included in this repository. Named `ALPHAT_MD.csv` in ADEPT. |
| `TC_MD.csv` | `TC` | Thermal conductivity | κ | W m⁻¹ K⁻¹ | same | 2249 | Non-equilibrium MD with hot/cold Langevin slabs at 320/280 K (mean 300 K). Isotropic amorphous-cell value. |
| `TG_MD.csv` | `TG` | Glass transition temperature | T<sub>g</sub> | K | same | 152 | Stepwise-cooling NPT dilatometry with an effective cooling rate of about 10 K ns⁻¹. See [Glass transition protocol](#glass-transition-protocol-tg_md). |
| `CP_GC.csv` | `CP` | Specific heat capacity | C<sub>p</sub> | J kg⁻¹ K⁻¹ | same | 12076 | Group-contribution estimate (Alosious et al., *Polym. Chem.* 2026, 17, 1028–1051). |
| `RHO_MD.csv` | `RHO` | Density | ρ | g cm⁻³ | same | 1935 | NPT at 300 K and 1 atm, averaged over the final 8 ns of equilibration. |
| `RG_MD.csv` | `RG` | Chain radius of gyration | R<sub>g</sub> | Å | same | 2499 | 300 K and 1 atm, averaged over all chains and the equilibrated trajectory. |
| `BULK_MD.csv` | `BULK` | Bulk modulus | K | GPa | same | 1011 | Symmetric ±2 % finite strains along six strain modes at 300 K, Voigt–Reuss–Hill average, averaged over 100 configurations from an 8 ns trajectory. Named `K_MD.csv` in ADEPT. |
| `SHEAR_MD.csv` | `SHEAR` | Shear modulus | G | GPa | same | 1006 | As for `BULK`. Named `G_MD.csv` in ADEPT. |
| `YOUNG_MD.csv` | `YOUNG` | Young's modulus | E | GPa | same | 1006 | E = 9KG / (3K + G). Named `E_MD.csv` in ADEPT. |
| `POISSON_MD.csv` | `POISSON` | Poisson's ratio | ν | dimensionless | same | 1006 | ν = (3K − 2G) / (2(3K + G)). Named `NU_MD.csv` in ADEPT. |
| `VISC_MD.csv` | `VISC` | Shear viscosity | η | 10⁻³ Pa s (mPa s) | `VISC × 1e-3` Pa s | 702 | Green–Kubo integral of the off-diagonal pressure autocorrelation over a 20 ps window, from a 5 ns NVT run at **T = T<sub>g</sub> + 50 K** (MD T<sub>g</sub> where available, otherwise an ML-estimated T<sub>g</sub>). The simulation temperature differs for each polymer and is not stored. See [Data-quality notes](#data-quality-notes). |
| `DIF_MD.csv` | `DIF` | Self-diffusion coefficient (all atoms) | D | 10⁻⁵ cm² s⁻¹ | `DIF × 1e-5` cm² s⁻¹ | 698 | Einstein relation: slope of the all-atom mean-square displacement over the same 5 ns NVT run as `VISC`, at **T = T<sub>g</sub> + 50 K**. Named `D_MD.csv` in ADEPT. See [Data-quality notes](#data-quality-notes). |
| `DC_MD.csv` | `DC` | Static dielectric constant (relative permittivity) | ε<sub>r</sub> | dimensionless | same | 739 | ε<sub>r</sub> = ε<sub>el</sub> + ε<sub>dip</sub> − 1. ε<sub>el</sub> = n² comes from the Lorentz–Lorenz relation, using the monomer polarizability (`ALPHA_DFT`) and the MD repeat-unit number density. ε<sub>dip</sub> = 1 + (⟨M²⟩ − ⟨M⟩²) / (3ε<sub>0</sub>k<sub>B</sub>TV) comes from total cell-dipole fluctuations in a 5 ns NVT run at 300 K. This is a zero-frequency estimate with no frequency or loss information. |
| `PE_MD.csv` | `PE` | Absolute permittivity | ε = ε<sub>r</sub>ε<sub>0</sub> | pF m⁻¹ (10⁻¹² F m⁻¹) | same | 739 | `PE = 8.854 × DC` exactly, so it carries the same information as `DC_MD`. It is **not** dimensionless and is **not** ε<sub>r</sub>. Do not treat `DC` and `PE` as independent targets. |
| `RI_MD.csv` | `RI` | Refractive index (electronic) | n | dimensionless | same | 739 | n = √ε<sub>el</sub> from the Lorentz–Lorenz relation (see `DC`). Electronic contribution only, with no specific wavelength. |
| `HOMO_DFT.csv` | `HOMO` | Monomer HOMO energy | E<sub>HOMO</sub> | eV | same | 2892 | ωB97M-D3BJ/6-311G(d,p) single point on the ωB97M-D3BJ/6-311+G(2d,p)-optimized monomer, with connection points capped by hydrogen. LanL2DZ is used for iodine. |
| `LUMO_DFT.csv` | `LUMO` | Monomer LUMO energy | E<sub>LUMO</sub> | eV | same | 2892 | As for `HOMO`. |
| `BANDGAP_DFT.csv` | `BANDGAP` | **Monomer HOMO–LUMO gap** (DFT) | ΔE<sub>HL</sub> | eV | same | 2892 | `BANDGAP = LUMO − HOMO` of the isolated capped monomer. This is **not** a polymer (periodic-chain or bulk) band gap or an optical gap, and its values (5.5–16.8 eV) are much larger than experimental polymer band gaps. |
| `MU_DFT.csv` | `MU` | Monomer dipole moment | μ | Debye | same | 2892 | As for `HOMO`. |
| `ALPHA_DFT.csv` | `ALPHA` | Monomer isotropic static polarizability | α | Å³ | same | 2018 | Finite-field method (±10⁻⁴ a.u.) at ωB97M-D3BJ/6-311+G(2d,p), using 6-311G(d,p) for Br-containing and LanL2DZ for I-containing monomers. α = Tr(**α**)/3. |
| `ETOTAL_DFT.csv` | `ETOTAL` | Monomer total electronic energy | E<sub>total</sub> | **kcal mol⁻¹** | same | 2892 | Same single point as `HOMO`. Not normalized by size, so it scales with the number of atoms. |

### Unit scaling in PolyGraphMT

Models are trained on the stored values. After inverse target scaling,
PolyGraphMT multiplies predictions for `td`, `dif` and `visc` by the factors in
`POST_SCALE_FACTORS` (`src/polygraphmt/utils.py`): `1e-7`, `1e-5` and `1e-3`
respectively. Exported predictions and `_orig` metrics for these three
properties are therefore in m² s⁻¹, cm² s⁻¹ and Pa s. All other properties are
reported in the stored units above.

## Glass transition protocol (`TG_MD`)

Every MD structure is first annealed: it is quenched from 1000 K to 300 K at
140 K ns⁻¹ and then equilibrated for 8 ns at 300 K and 1 atm. **This 140 K ns⁻¹
annealing quench is not the cooling rate used to determine T<sub>g</sub>.**

T<sub>g</sub> is then obtained from a separate stepwise-cooling scan that
starts from the equilibrated 300 K structure:

1. An ML model trained on experimental T<sub>g</sub> provides an initial
   estimate, T<sub>ML</sub>, which is used only to place the scan window.
2. The system is cooled under NPT (1 atm, 1 fs time step) in 10 K steps from
   T<sub>ML</sub> + 150 K down to T<sub>ML</sub> − 150 K (but not below 50 K).
   Each step continues from the end of the previous one.
3. At each temperature the system is equilibrated for 0.5 ns, and the density
   is averaged over a further 0.5 ns production segment. This corresponds to
   an effective cooling rate of **10 K per 1 ns ≈ 10 K ns⁻¹**.
4. Straight lines are fitted to the lowest and highest 5, 6, 7 and 8
   density–temperature points. T<sub>g</sub> is the mean of the four
   intersection temperatures. A result is discarded if any intersection falls
   outside the scan window.

Because the effective rate is about 10¹⁰–10¹¹ times faster than a typical DSC
scan (10 K min⁻¹), simulated T<sub>g</sub> values are expected to be biased
upward relative to experiment.

## File names in ADEPT vs PolyGraphMT

The ADEPT workflow writes some properties under different file names. They
refer to the same quantities:

| ADEPT (`POLYMER_DATA/RESULTS/`) | PolyGraphMT (`data/raw/`) |
| --- | --- |
| `ALPHAT_MD.csv` | `TD_MD.csv` |
| `D_MD.csv` | `DIF_MD.csv` |
| `K_MD.csv` | `BULK_MD.csv` |
| `G_MD.csv` | `SHEAR_MD.csv` |
| `E_MD.csv` | `YOUNG_MD.csv` |
| `NU_MD.csv` | `POISSON_MD.csv` |

All other file names are the same in both repositories. The PolyGraphMT names
are kept unchanged because the released checkpoints under `models/` use them as
task keys (`td`, `dif`, `bulk`, …).

## Data-quality notes

The values below are left in the files exactly as generated. They are listed
here so that users can decide whether to exclude them.

**`DC_MD` (and the corresponding `PE_MD` rows).** Nine polymers have
ε<sub>r</sub> > 4.82 (above Q3 + 3·IQR of the distribution). Each dielectric
value comes from a single 5 ns trajectory with no convergence check on the
dipole-fluctuation term, so these values are unverified. Some may be genuine,
for example strongly polar, flexible side chains such as cyanoethyl groups,
but `*CS*` in particular is likely unconverged. Treat them with caution in
quantitative analyses.

| SMILES | `DC` |
| --- | --- |
| `*CS*` | 12.78 |
| `*CC(*)C(=O)OCCSCC#N` | 8.89 |
| `*Cc1ccc(CSSS*)cc1` | 7.17 |
| `*CC(*)OCCCl` | 6.48 |
| `*C/C=C/CS(*)(=O)=O` | 6.08 |
| `*CC(*)C(=O)NCCO` | 5.42 |
| `*CC(*)C(=O)OCCCCCCSCC#N` | 5.22 |
| `*CC(*)c1nc(N)nc(N)n1` | 5.20 |
| `*CC(*)C(=O)OCCCSCCC#N` | 5.10 |

**`RHO_MD`.** Nine structures have ρ < 0.5 g cm⁻³, which means the simulation
cell did not condense:
`*CC(*)OC(=O)c1cccc(Br)c1`, `*CC(CBr)O*`, `*C(F)(F)C1(F)C(F)(F)C(*)(F)C1(F)F`,
`*C(F)(F)C1(F)C(F)(F)C(*)(F)C(F)(F)C1(F)F`, `*CC(*)C(=O)c1ccc(Br)c(OC)c1`,
`*C1OC(C(F)(F)F)(C(F)(F)F)OC1(*)F`, `*C1(F)OC(C(F)(F)F)(C(F)(F)Cl)OC1(*)F`,
`*CC(*)c1ccc(C(=O)CCC)cc1`, `*CC(*)(C#N)C(=O)OCCCCCC`.
Other MD properties computed from these structures should also be treated with
caution. They appear in `RG_MD` (9), `DC_MD`/`PE_MD`/`RI_MD` (6), `VISC_MD` (3),
`DIF_MD` (2), and `TC_MD`, `TD_MD` and the mechanical files (1 each).

**`ALPHA_DFT`.** The value for `*S*` (1.923 Å³) is about half of what is
expected for this monomer and is likely a finite-field artifact.

**`DIF_MD`.** More than half of the values exceed 10⁻⁴ cm² s⁻¹, which is not
physically plausible for atomic self-diffusion in a dense polymer. The MSD slope
is fitted from t = 0, so it includes the non-diffusive short-time regime, and
centre-of-mass drift is not removed. These values are not recommended for
quantitative use.

**`VISC_MD`.** The Green–Kubo integral is truncated at 20 ps, which is far
shorter than polymer stress-relaxation times. The values are also computed at
a different temperature (T<sub>g</sub> + 50 K) for each polymer. They are
therefore not comparable with experimental melt or zero-shear viscosities and
should be used only for relative comparisons.

**General.** All MD properties are simulation-level (computational-fidelity)
values for isotropic amorphous cells of about 3600 atoms (six chains of about
600 atoms). Crystallinity, chain orientation, molecular weight and
polydispersity are not represented.

## Supported naming conventions

- `<target>_<fid>.csv`
- `<fid>/<target>.csv`

Keep generated artifacts, train/val/test splits, and result folders outside `data/raw/`.
