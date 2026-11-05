# Single-basin reproduction log

Baseline 57d9f21 preserves the supplied source (not verified against GitHub tag).
Basin 01022500 occurs in the supplied 531-row results table; full paper-subset identity is not yet established.
CAMELS basin mean meteorological forcing v1p2, Daymet, selected explicitly. The paper does not name a forcing product in its searchable text, so this is a conditional code reproduction, not a verified exact numerical reproduction.

Faithful settings: 1980-1999 train, 2000-2003 validation, 2004-2008 test; sequence 365; epochs 60; seed 42; batch 64; original model, optimizer, losses, scheduler and metrics.
Adaptations: native CAMELS forcing/flow filenames; physical-attributes filename; string basin IDs preserving leading zeros; fail on missing/nonfinite static fields; reject ambiguous forcing. Separate isolated Daymet data directory. Four CPU threads for local resource use. Instrumentation only otherwise.

Data audit: 12784 forcing/flow days, 1980-2014. No negative discharge in 1980-2008; finite meteorology. Physical vector [1,620.38,92.68,17.79072,0.9232]. Forcing header area 587.675987 km2 differs from static 620.38 km2; retained author's conversion using forcing header, not overwritten. Forest field labelled percent but values are fractions; retained raw values.

Unchanged source caveats: sequences are constructed separately inside each split, losing the first 364 target days of validation/test, and __len__ omits the last possible sample; evaluate_model exponentiates log(Q+0.01), so its observations are Q+0.01. Source features order [Prcp,Srad,Tmax,Tmin,Vp,Swe]. Do not silently fix these while comparing with supplied code.

Scientific boundaries: equation (6) examines proportional response to precipitation scaling, not additivity of two storms. EUH is a model impulse-derived curve, not observed unit hydrograph. Raw meteorology predicts total streamflow without explicit effective rainfall, direct runoff or baseflow separation. No Mamba trainer found in supplied Python code; Mamba table entries are published reference results only.

Published reference for 01022500: HEN NSE 0.7777571428, KGE 0.6628778742, Pearson r 0.9146271316. Matching these is not guaranteed by the available source/data provenance.
