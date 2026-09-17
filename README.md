# WHO GBCI trajectory: analysis code

Analysis code for an uncertainty-aware assessment of whether countries are on track for the
WHO Global Breast Cancer Initiative benchmark of a **2.5% annual reduction** in female breast
cancer mortality, across 204 countries and territories, 2015 to 2023.

**Almost no country was on trajectory, and age-standardised mortality was rising.** Of the 204
countries assessed, 2 were on trajectory, 5 uncertain and 197 off trajectory. The all-country
mean annual percentage change was **+0.59% per year** (95% CI +0.28 to +0.89). Only the
European Region had a declining mean (−1.08%, −1.45 to −0.71), and every on-trajectory or
uncertain country lay within it. Annual deaths rose from 588,042 in 2015 to 763,871 in 2023.

The methodological point is that **data-quality strata separated as strongly as geography**:
the widest tertile of GBD uncertainty averaged +3.16% per year against −0.69% in the narrowest.
Geographical and income gradients align with measurement-quality gradients and should not be
read as purely epidemiological. Strengthening registration is a precondition for monitoring the
benchmark, not only for meeting it.

Progress is usually reported as point estimates or league-table ranks, neither of which
separates a country that is genuinely improving from one whose data are too sparse to reveal
any trend. This code does that separation explicitly: slopes are fitted by weighted least
squares, weighting each annual estimate by the inverse of its measurement variance, then
partially pooled towards regional means by empirical Bayes. Countries are classified on
trajectory at a posterior probability of 0.80 or above, uncertain between 0.20 and 0.79, and
off trajectory below 0.20.

## What is in this repository

Only the analysis code, under `04_code/`.

```
04_code/
  run_pipeline.py                 single entry point: download, process, analyse, report
  study.py                        core analysis: retrieval, panel construction, slope
                                  estimation, empirical Bayes pooling, classification
  audit_survival_circularity.py   pre-specified audit testing whether WHO survival estimates
                                  can serve as a predictor, or whether they share too much
                                  variance with the outcome to be used
  manuscript_numbers.py           prints every reported quantity from the machine-readable
                                  results, so any number can be traced to its source
  tests/test_study.py             unit tests
```

## What is not in this repository, and why

The data directories are absent by design. Raw source archives are third-party and are not
redistributed here, and the derived panels, results, figures and manuscript drafts are not
published either.

This matters for running the code. The scripts resolve the project root as the parent of
`04_code/` and expect these sibling directories:

```
01_metadata/  02_raw_data/  03_processed_data/
05_results/   06_figures/   07_tables/   09_logs/
```

`run_pipeline.py download` retrieves the public sources into `02_raw_data/`. Sources are the
Global Burden of Disease 2023 release, the WHO Mortality Database, WHO Global Health Estimates
and the WHO Global Health Observatory. No endpoint used here requires credentials.

## Running it

```bash
python -m pip install -r requirements.txt

python 04_code/run_pipeline.py download   # retrieve public sources
python 04_code/run_pipeline.py process    # build the analytic panel
python 04_code/run_pipeline.py analyse    # slopes, pooling, classification
python 04_code/run_pipeline.py report     # results and figures
python 04_code/run_pipeline.py all

python -m pytest 04_code/tests -q
```

## Citation

This code accompanies a manuscript under submission. Please cite the article once published.

## Licence

MIT. See [LICENSE](LICENSE).

This covers the code in this repository only. The data sources it retrieves are third-party
and carry their own terms; see the provider in each case.
