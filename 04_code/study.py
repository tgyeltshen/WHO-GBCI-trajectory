"""Core functions for Project 3.

All raw downloads are append-only. Derived files may be regenerated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_raw_data"
PROCESSED = ROOT / "03_processed_data"
RESULTS = ROOT / "05_results"
FIGURES = ROOT / "06_figures"
TABLES = ROOT / "07_tables"
LOGS = ROOT / "09_logs"
METADATA = ROOT / "01_metadata"
ACCESS_DATE = datetime.now(timezone.utc).date().isoformat()
SEED = 20260813

WHO_BASE = "https://ghoapi.azureedge.net/api"
WHO_INDICATORS = {
    "SA_0000001438": "who_mortality",
    "CANCERSURVIVAL_BREASTCANCER": "who_survival",
    # UHC service coverage index. The World Bank mirror SH.UHC.SRVS.CV.XD is archived and its API
    # now returns "The indicator was not found", so the index is taken from its WHO origin instead
    # (deviation D2, 2026-08-16). WHO serves 2000-2023 for ~195 countries.
    "UHC_INDEX_REPORTED": "who_uhc_index",
}
WB_INDICATORS = {
    "SH.XPD.OOPC.CH.ZS": "oop_share_che",
    "SH.XPD.GHED.CH.ZS": "government_share_che",
    "SH.XPD.CHEX.PC.CD": "che_per_capita_usd",
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp_constant",
}
WHO_MDB_FILES = {
    "documentation.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/mort_documentation71f9e29d-7e3f-41e6-aafc-c4c1775c7aa3.zip?sfvrsn=40cce9be_42",
    "availability.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/mort_availability.zip?sfvrsn=23261e11_33",
    "country_codes.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/mort_country_codes.zip?sfvrsn=800faac2_5",
    "population.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/mort_pop.zip?sfvrsn=937039fc_26",
    "morticd10_part4_2013_2016.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/morticd10_part4.zip?sfvrsn=259c5c23_30",
    "morticd10_part5_2017_2020.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/morticd10_part5.zip?sfvrsn=ad970d0b_34",
    "morticd10_part6_2021_onwards.zip": "https://cdn.who.int/media/docs/default-source/world-health-data-platform/mortality-raw-data/morticd10_part6.zip?sfvrsn=ec801a61_4",
}
WHO_GHE_ASDR_URL = "https://cdn.who.int/media/docs/default-source/gho-documents/global-health-estimates/ghe2021_deathrates_bycountry_asdr.xlsx?sfvrsn=d2ead5e5_3"
# Female population by five-year age group, 1950-2100, medium variant. Needed to generate deaths
# under the 2040 scenarios and to separate population size and age structure in the decomposition.
# The five-year grouping matches the GBD age bands; the file carries ISO3 codes directly.
UN_WPP_FILES = {
    "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz": (
        "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
        "CSV_FILES/WPP2024_PopulationByAge5GroupSex_Medium.csv.gz"
    ),
}
# WHO world standard population, 18 five-year bands from 0-4 to 85+. The published percentages are
# rounded to two decimals and sum to 100.03, not 100, so they are normalised by their own sum
# rather than by 100. Dividing by 100 inflates every standardised rate by 0.03%. Annual percent
# change is invariant to a constant multiplicative factor, so trend outputs are unaffected.
# Country boundaries for the trajectory map. Natural Earth is public domain and versioned in a
# public repository, so the exact geometry behind a published figure stays retrievable. The 1:110m
# scale carries 177 polygons, so microstates analysed in the study are not drawable; the figure
# states how many analysed countries it shows.
NATURAL_EARTH_FILES = {
    "ne_110m_admin_0_countries.geojson": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
        "ne_110m_admin_0_countries.geojson"
    ),
}
WHO_STANDARD_WEIGHTS = np.array(
    [8.86, 8.69, 8.60, 8.47, 8.22, 7.93, 7.61, 7.15, 6.59, 6.04, 5.37, 4.55, 3.72, 2.96, 2.21, 1.52, 0.91, 0.63]
)
WHO_STANDARD_WEIGHTS = WHO_STANDARD_WEIGHTS / WHO_STANDARD_WEIGHTS.sum()


def ensure_directories() -> None:
    for path in (RAW, PROCESSED, RESULTS, FIGURES, TABLES, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_once(path: Path, content: bytes) -> str:
    """Write immutable raw bytes, or verify an already-present identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = sha256(path)
        incoming = hashlib.sha256(content).hexdigest()
        if existing != incoming:
            raise RuntimeError(f"Refusing to overwrite raw file with different bytes: {path}")
        return "verified_existing"
    path.write_bytes(content)
    return "downloaded"


def get_bytes(url: str, *, timeout: int = 120, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Project3-reproducible-research/1.0"},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def download_who_indicator(code: str) -> list[Path]:
    target_dir = RAW / "who_gho" / ACCESS_DATE / code
    url = f"{WHO_BASE}/{code}?%24top=1000"
    paths: list[Path] = []
    page = 1
    while url:
        content = get_bytes(url)
        path = target_dir / f"page_{page:03d}.json"
        write_bytes_once(path, content)
        paths.append(path)
        payload = json.loads(content)
        url = payload.get("@odata.nextLink")
        page += 1
    return paths


def download_world_bank_indicator(code: str) -> Path:
    safe = code.replace(".", "_")
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{code}"
        "?format=json&per_page=30000&date=1990:2025"
    )
    path = RAW / "world_bank" / ACCESS_DATE / f"{safe}.json"
    write_bytes_once(path, get_bytes(url))
    return path


def download_world_bank_countries() -> Path:
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    path = RAW / "world_bank" / ACCESS_DATE / "country_metadata.json"
    write_bytes_once(path, get_bytes(url))
    return path


def download_named_public_file(source: str, filename: str, url: str) -> Path:
    path = RAW / source / ACCESS_DATE / filename
    write_bytes_once(path, get_bytes(url, timeout=300))
    return path


def download_public_data() -> list[Path]:
    ensure_directories()
    paths: list[Path] = []
    failures = []
    for code in WHO_INDICATORS:
        try:
            paths.extend(download_who_indicator(code))
        except requests.RequestException as exc:
            failures.append({"source": "WHO GHO", "code": code, "error": str(exc)})
    for code in WB_INDICATORS:
        try:
            paths.append(download_world_bank_indicator(code))
        except requests.RequestException as exc:
            failures.append({"source": "World Bank", "code": code, "error": str(exc)})
    try:
        paths.append(download_world_bank_countries())
    except requests.RequestException as exc:
        failures.append({"source": "World Bank", "code": "country_metadata", "error": str(exc)})
    for filename, url in UN_WPP_FILES.items():
        try:
            paths.append(download_named_public_file("un_wpp", filename, url))
        except requests.RequestException as exc:
            failures.append({"source": "UN WPP", "code": filename, "error": str(exc)})
    for filename, url in WHO_MDB_FILES.items():
        try:
            paths.append(download_named_public_file("who_mortality_database", filename, url))
        except requests.RequestException as exc:
            failures.append({"source": "WHO Mortality Database", "code": filename, "error": str(exc)})
    for filename, url in NATURAL_EARTH_FILES.items():
        try:
            paths.append(download_named_public_file("natural_earth", filename, url))
        except requests.RequestException as exc:
            failures.append({"source": "Natural Earth", "code": filename, "error": str(exc)})
    try:
        paths.append(download_named_public_file("who_ghe", "ghe2021_deathrates_bycountry_asdr.xlsx", WHO_GHE_ASDR_URL))
    except requests.RequestException as exc:
        failures.append({"source": "WHO Global Health Estimates", "code": "ghe2021_asdr", "error": str(exc)})
    (LOGS / f"download_failures_{ACCESS_DATE}.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    build_source_manifest()
    if len(paths) == 0:
        raise RuntimeError("No public source downloads succeeded")
    return paths


def source_attributes(path: Path) -> dict[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    if "/who_gho/" in f"/{rel}":
        code = path.parent.name
        return {
            "source": "WHO Global Health Observatory OData API",
            "release": f"live snapshot {path.parents[1].name}",
            "url_or_route": f"{WHO_BASE}/{code}",
            "license_or_terms": "https://www.who.int/about/policies/publishing/copyright",
            "data_status": "modelled or mixed; see indicator metadata",
        }
    if "/world_bank/" in f"/{rel}":
        stem = path.stem.replace("_", ".")
        return {
            "source": "World Bank Indicators API",
            "release": f"live snapshot {path.parent.name}",
            "url_or_route": f"https://api.worldbank.org/v2/country/all/indicator/{stem}",
            "license_or_terms": "https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets",
            "data_status": "country-reported/modelled as indicator metadata specifies",
        }
    if "/who_mortality_database/" in f"/{rel}":
        return {
            "source": "WHO Mortality Database",
            "release": "raw files updated 2026-02-23",
            "url_or_route": "https://www.who.int/data/data-collection-tools/who-mortality-database",
            "license_or_terms": "WHO Mortality Database non-commercial data-use guidelines",
            "data_status": "country-reported observed vital registration",
        }
    if "/who_ghe/" in f"/{rel}":
        return {
            "source": "WHO Global Health Estimates 2021",
            "release": "GHE 2021 (published 2024)",
            "url_or_route": WHO_GHE_ASDR_URL,
            "license_or_terms": "https://www.who.int/about/policies/publishing/copyright",
            "data_status": "modelled",
        }
    if "/ihme_gbd/" in f"/{rel}":
        return {
            "source": "IHME GBD Results Tool",
            "release": "GBD 2023",
            "url_or_route": "https://vizhub.healthdata.org/gbd-results/",
            "license_or_terms": "IHME free-of-charge non-commercial user agreement",
            "data_status": "modelled",
        }
    if "/un_wpp/" in f"/{rel}":
        return {
            "source": "UN World Population Prospects 2024",
            "release": "WPP 2024, medium variant",
            "url_or_route": UN_WPP_FILES.get(path.name, "https://population.un.org/wpp/"),
            "license_or_terms": "https://population.un.org/wpp/ (CC BY 3.0 IGO)",
            "data_status": "estimates to 2023; medium-variant projections thereafter",
        }
    if "/natural_earth/" in f"/{rel}":
        return {
            "source": "Natural Earth vector, 1:110m admin 0 countries",
            "release": "natural-earth-vector repository, main branch",
            "url_or_route": NATURAL_EARTH_FILES.get(path.name, "https://www.naturalearthdata.com/"),
            "license_or_terms": "public domain (Natural Earth terms of use)",
            "data_status": "cartographic boundaries; display only, not analytical input",
        }
    return {
        "source": "unclassified",
        "release": "unknown",
        "url_or_route": "unknown",
        "license_or_terms": "unknown",
        "data_status": "unknown",
    }


def build_source_manifest() -> pd.DataFrame:
    ensure_directories()
    rows = []
    for path in sorted(p for p in RAW.rglob("*") if p.is_file()):
        attr = source_attributes(path)
        raw_parts = path.relative_to(RAW).parts
        recorded_access_date = raw_parts[1] if len(raw_parts) > 1 else "unknown"
        rows.append(
            {
                "source": attr["source"],
                "release": attr["release"],
                "relative_path": path.relative_to(ROOT).as_posix(),
                "url_or_route": attr["url_or_route"],
                "access_date": recorded_access_date if attr["source"] != "IHME GBD Results Tool" else "manual",
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license_or_terms": attr["license_or_terms"],
                "data_status": attr["data_status"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(METADATA / "source_manifest.csv", index=False)
    return frame


def latest_snapshot_dir(source: str) -> Path | None:
    base = RAW / source
    if not base.exists():
        return None
    dirs = sorted(p for p in base.iterdir() if p.is_dir())
    return dirs[-1] if dirs else None


def read_odata_pages(indicator: str) -> pd.DataFrame:
    base = latest_snapshot_dir("who_gho")
    if base is None:
        return pd.DataFrame()
    directory = base / indicator
    rows: list[dict] = []
    for path in sorted(directory.glob("page_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("value", []))
    return pd.DataFrame(rows)


def process_who() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for indicator, name in WHO_INDICATORS.items():
        frame = read_odata_pages(indicator)
        if frame.empty:
            continue
        frame.insert(0, "source_status", "modelled" if name == "who_mortality" else "mixed_modelled_observed")
        frame.to_csv(PROCESSED / f"{name}.csv", index=False)
        outputs[name] = frame
    return outputs


def _read_first_zip_member(path: Path, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            return pd.read_csv(handle, **kwargs)


def process_who_ghe_asdr() -> pd.DataFrame:
    base = latest_snapshot_dir("who_ghe")
    if base is None:
        return pd.DataFrame()
    path = base / "ghe2021_deathrates_bycountry_asdr.xlsx"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        match = re.fullmatch(r"ASDR (\d{4})", sheet)
        if not match:
            continue
        year = int(match.group(1))
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        country_names = frame.iloc[6, 7:].tolist()
        iso3_codes = frame.iloc[7, 7:].tolist()
        candidate_rows = []
        for idx in frame.index:
            sex = str(frame.iloc[idx, 0]).strip().casefold()
            cause = str(frame.iloc[idx, 5]).strip().casefold()
            if sex == "females" and cause == "breast cancer":
                candidate_rows.append(idx)
        if len(candidate_rows) != 1:
            raise RuntimeError(f"Expected one female breast-cancer row in {sheet}, found {candidate_rows}")
        values = frame.iloc[candidate_rows[0], 7:].tolist()
        for country, iso3, value in zip(country_names, iso3_codes, values):
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric) or pd.isna(iso3):
                continue
            rows.append(
                {
                    "iso3": str(iso3),
                    "country": str(country),
                    "year": year,
                    "asdr_per_100k": float(numeric),
                    "source_status": "WHO_GHE_modelled",
                    "standard_population": "WHO GHE 2021 workbook standard",
                }
            )
    result = pd.DataFrame(rows).sort_values(["iso3", "year"])
    result.to_csv(PROCESSED / "who_ghe_breast_cancer_asdr.csv", index=False)
    return result


def _breast_cancer_mask(chunk: pd.DataFrame) -> pd.Series:
    list_code = chunk["List"].astype(str).str.strip()
    cause = chunk["Cause"].astype(str).str.strip().str.upper()
    return (
        (list_code.eq("101") & cause.eq("1036"))
        | (list_code.eq("103") & cause.eq("C50"))
        | (list_code.isin(["104", "10M"]) & cause.str.startswith("C50"))
    )


def _national_rows(frame: pd.DataFrame) -> pd.Series:
    return frame["Admin1"].isna() & frame["SubDiv"].isna()


def _age_standardised_rate(deaths: pd.Series, population: pd.Series, age_format: str) -> float:
    """Direct standardisation for MDB formats 00/01/02 using WHO weights."""
    if str(age_format).zfill(2) not in {"00", "01", "02"}:
        return float("nan")
    d = pd.to_numeric(deaths, errors="coerce")
    p = pd.to_numeric(population, errors="coerce")
    death_groups = [d[[f"Deaths{i}" for i in range(2, 7)]].sum(min_count=5)]
    pop_groups = [p[[f"Pop{i}" for i in range(2, 7)]].sum(min_count=5)]
    for index in range(7, 23):
        death_groups.append(d.get(f"Deaths{index}", np.nan))
        pop_groups.append(p.get(f"Pop{index}", np.nan))
    death_groups.append(d[["Deaths23", "Deaths24", "Deaths25"]].sum(min_count=1))
    pop_groups.append(p[["Pop23", "Pop24", "Pop25"]].sum(min_count=1))
    death_groups = np.asarray(death_groups, dtype=float)
    pop_groups = np.asarray(pop_groups, dtype=float)
    if len(death_groups) != len(WHO_STANDARD_WEIGHTS) or np.any(~np.isfinite(pop_groups)) or np.any(pop_groups <= 0):
        return float("nan")
    rates = death_groups / pop_groups * 100000.0
    return float(np.sum(rates * WHO_STANDARD_WEIGHTS))


def process_who_mortality_database() -> pd.DataFrame:
    base = latest_snapshot_dir("who_mortality_database")
    if base is None:
        return pd.DataFrame()
    death_files = sorted(base.glob("morticd10_part*.zip"))
    population_path = base / "population.zip"
    country_path = base / "country_codes.zip"
    if not death_files or not population_path.exists() or not country_path.exists():
        return pd.DataFrame()
    kept = []
    string_columns = {"Country": str, "Admin1": str, "SubDiv": str, "List": str, "Cause": str, "Frmat": str}
    for path in death_files:
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                for chunk in pd.read_csv(handle, dtype=string_columns, chunksize=150000, low_memory=False):
                    mask = _national_rows(chunk) & chunk["Sex"].eq(2) & _breast_cancer_mask(chunk)
                    if mask.any():
                        kept.append(chunk.loc[mask].copy())
    if not kept:
        return pd.DataFrame()
    deaths = pd.concat(kept, ignore_index=True)
    death_columns = [f"Deaths{i}" for i in range(1, 26)]
    for column in death_columns:
        deaths[column] = pd.to_numeric(deaths[column], errors="coerce")
    grouped = deaths.groupby(["Country", "Year", "List", "Frmat"], as_index=False)[death_columns].sum(min_count=1)
    priority = {"101": 1, "103": 2, "104": 3, "10M": 4}
    grouped["list_priority"] = grouped["List"].map(priority).fillna(0)
    grouped = grouped.sort_values("list_priority").drop_duplicates(["Country", "Year"], keep="last")

    population = _read_first_zip_member(population_path, dtype={"Country": str, "Admin1": str, "SubDiv": str, "Frmat": str}, low_memory=False)
    population = population[_national_rows(population) & population["Sex"].eq(2)].copy()
    pop_columns = [f"Pop{i}" for i in range(1, 26)]
    for column in pop_columns:
        population[column] = pd.to_numeric(population[column], errors="coerce")
    population = population.drop_duplicates(["Country", "Year", "Frmat"])
    merged = grouped.merge(population[["Country", "Year", "Frmat"] + pop_columns], on=["Country", "Year", "Frmat"], how="left", indicator=True)

    country_codes = _read_first_zip_member(country_path, dtype={"country": str})
    country_codes.columns = [normalise_column(c) for c in country_codes.columns]
    name_map = dict(zip(country_codes["country"], country_codes["name"]))
    records = []
    for _, row in merged.iterrows():
        asdr = _age_standardised_rate(row[death_columns], row[pop_columns], row["Frmat"]) if row["_merge"] == "both" else float("nan")
        records.append(
            {
                "who_country_code": row["Country"],
                "country": name_map.get(str(row["Country"]), "unmapped"),
                "year": int(row["Year"]),
                "list": row["List"],
                "age_format": str(row["Frmat"]).zfill(2),
                "female_deaths_all_ages": row["Deaths1"],
                "asdr_per_100k": asdr,
                "population_match": row["_merge"],
                "source_status": "observed_country_reported",
                "standard_population": "WHO world standard 2000-2025",
            }
        )
    result = pd.DataFrame(records).sort_values(["country", "year"])
    result.to_csv(PROCESSED / "who_mdb_breast_cancer_observed.csv", index=False)
    return result


def process_world_bank() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = latest_snapshot_dir("world_bank")
    if base is None:
        return pd.DataFrame(), pd.DataFrame()
    long_rows = []
    for code, short_name in WB_INDICATORS.items():
        path = base / f"{code.replace('.', '_')}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        # A retired or mistyped indicator code returns HTTP 200 with an error message and no data.
        # Writing that to disk and skipping it silently drops a pre-specified variable, so refuse.
        header = payload[0] if isinstance(payload, list) and payload else {}
        if isinstance(header, dict) and header.get("message"):
            detail = header["message"][0].get("value") if header["message"] else "unknown error"
            raise ValueError(
                f"World Bank indicator {code} returned an API error instead of data: {detail}. "
                "Update the indicator code or move the variable to another source; the pipeline "
                "will not treat an error payload as missing data."
            )
        data = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        for row in data or []:
            if row.get("value") is None:
                continue
            long_rows.append(
                {
                    "iso3": row.get("countryiso3code"),
                    "country": (row.get("country") or {}).get("value"),
                    "year": int(row["date"]),
                    "indicator_code": code,
                    "indicator": short_name,
                    "value": float(row["value"]),
                    "source_status": "country_reported_or_modelled",
                }
            )
    long = pd.DataFrame(long_rows)
    if not long.empty:
        long.to_csv(PROCESSED / "world_bank_indicators_long.csv", index=False)
        wide = long.pivot_table(index=["iso3", "country", "year"], columns="indicator", values="value").reset_index()
        wide.columns.name = None
        wide.to_csv(PROCESSED / "world_bank_indicators_wide.csv", index=False)
    else:
        wide = pd.DataFrame()

    country_path = base / "country_metadata.json"
    crosswalk_rows = []
    if country_path.exists():
        payload = json.loads(country_path.read_text(encoding="utf-8"))
        data = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        for row in data or []:
            crosswalk_rows.append(
                {
                    "iso3": row.get("id"),
                    "iso2": row.get("iso2Code"),
                    "country": row.get("name"),
                    "region": (row.get("region") or {}).get("value"),
                    "income_group": (row.get("incomeLevel") or {}).get("value"),
                    "is_aggregate": (row.get("region") or {}).get("id") == "NA",
                    "decision": "exclude" if (row.get("region") or {}).get("id") == "NA" else "include",
                }
            )
    crosswalk = pd.DataFrame(crosswalk_rows)
    if not crosswalk.empty:
        crosswalk.to_csv(METADATA / "country_crosswalk.csv", index=False)
    return long, crosswalk


def normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def csv_frames_from_gbd() -> Iterable[tuple[str, pd.DataFrame]]:
    directory = RAW / "ihme_gbd" / "gbd_2023"
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".csv":
            yield path.name, pd.read_csv(path)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.lower().endswith(".csv"):
                        with archive.open(member) as handle:
                            yield f"{path.name}:{member}", pd.read_csv(handle)


def first_column(columns: Iterable[str], choices: Iterable[str]) -> str | None:
    column_set = set(columns)
    return next((name for name in choices if name in column_set), None)


# GBD 2023 aggregate rows for the six WHO regions. A country-level trajectory analysis
# must not treat these as countries: they would inflate the denominator and mix a
# population-weighted regional average into a distribution of national trends. The export
# specification permits requesting regions, so they are separated here rather than assumed absent.
WHO_REGION_LOCATION_IDS = frozenset(
    {
        44563,  # African Region
        44564,  # Region of the Americas
        44565,  # South-East Asia Region
        44566,  # European Region
        44567,  # Eastern Mediterranean Region
        44568,  # Western Pacific Region
        479,  # exported under the bare label "WHO region"; 1990-2003 only, not a country
    }
)

# GBD location names that differ from the World Bank spelling. Resolved by hand against the
# World Bank country list rather than by fuzzy matching, because a wrong automatic match would
# silently attach one country's health-system series to another country's mortality.
GBD_NAME_TO_ISO3 = {
    "Bahamas": "BHS",
    "Bolivia (Plurinational State of)": "BOL",
    "Congo": "COG",
    "Cook Islands": "COK",
    "Côte d'Ivoire": "CIV",
    "Democratic People's Republic of Korea": "PRK",
    "Democratic Republic of the Congo": "COD",
    "Egypt": "EGY",
    "Gambia": "GMB",
    "Iran (Islamic Republic of)": "IRN",
    "Kyrgyzstan": "KGZ",
    "Lao People's Democratic Republic": "LAO",
    "Micronesia (Federated States of)": "FSM",
    "Nauru": "NRU",
    "Niue": "NIU",
    "Palestine": "PSE",
    "Puerto Rico": "PRI",
    "Republic of Korea": "KOR",
    "Republic of Moldova": "MDA",
    "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Slovakia": "SVK",
    "Somalia": "SOM",
    "Taiwan": "TWN",
    "Tokelau": "TKL",
    "Türkiye": "TUR",
    "United Republic of Tanzania": "TZA",
    "United States Virgin Islands": "VIR",
    "United States of America": "USA",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Yemen": "YEM",
}


def canonicalise_gbd(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    frame = frame.rename(columns={c: normalise_column(c) for c in frame.columns}).copy()
    loc = first_column(frame.columns, ["location_name", "location", "location_id"])
    year = first_column(frame.columns, ["year", "year_id"])
    val = first_column(frame.columns, ["val", "value", "mean", "rate"])
    low = first_column(frame.columns, ["lower", "low", "lower_bound"])
    high = first_column(frame.columns, ["upper", "high", "upper_bound"])
    required = {"location": loc, "year": year, "value": val, "lower": low, "upper": high}
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"GBD file {source_file} lacks canonical fields: {missing}; columns={list(frame.columns)}")

    mask = pd.Series(True, index=frame.index)
    filters = {
        "cause_name": "breast cancer",
        "measure_name": "deaths",
        "metric_name": "rate",
        "age_name": "age-standardized",
        "sex_name": "female",
    }
    for column, expected in filters.items():
        if column in frame:
            mask &= frame[column].astype(str).str.casefold().eq(expected.casefold())
    out = pd.DataFrame(
        {
            "location": frame.loc[mask, loc].astype(str),
            "year": pd.to_numeric(frame.loc[mask, year], errors="coerce"),
            "rate": pd.to_numeric(frame.loc[mask, val], errors="coerce"),
            "lower": pd.to_numeric(frame.loc[mask, low], errors="coerce"),
            "upper": pd.to_numeric(frame.loc[mask, high], errors="coerce"),
            "source_file": source_file,
            "source_status": "modelled",
        }
    ).dropna(subset=["year", "rate", "lower", "upper"])
    out["year"] = out["year"].astype(int)
    # Carry the numeric location id so aggregates can be separated by identifier rather than
    # by matching display names, which vary between releases and translations.
    if "location_id" in frame.columns:
        out["location_id"] = pd.to_numeric(frame.loc[mask, "location_id"], errors="coerce")
    else:
        out["location_id"] = pd.NA
    return out


def world_bank_name_to_iso3() -> dict[str, str]:
    """Country name to ISO3 from the World Bank metadata snapshot, aggregates excluded."""
    base = latest_snapshot_dir("world_bank")
    if base is None:
        return {}
    path = base / "country_metadata.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    # region id "NA" marks an aggregate row such as "World" or "High income".
    return {
        row["name"]: row["id"]
        for row in rows or []
        if (row.get("region") or {}).get("id") != "NA" and row.get("id")
    }


def gbd_locations_to_iso3(locations: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Map GBD location names to ISO3. Returns the mapping and any names left unresolved."""
    lookup = world_bank_name_to_iso3()
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for name in sorted(set(locations)):
        if name in GBD_NAME_TO_ISO3:
            mapping[name] = GBD_NAME_TO_ISO3[name]
        elif name in lookup:
            mapping[name] = lookup[name]
        else:
            unresolved.append(name)
    return mapping, unresolved


def is_who_region_aggregate(frame: pd.DataFrame) -> pd.Series:
    """True for rows that are WHO regional aggregates rather than countries or territories."""
    if "location_id" in frame.columns:
        by_id = pd.to_numeric(frame["location_id"], errors="coerce").isin(WHO_REGION_LOCATION_IDS)
    else:
        by_id = pd.Series(False, index=frame.index)
    # Name fallback for exports that omit the id column entirely.
    by_name = frame["location"].astype(str).str.strip().isin(
        {
            "African Region",
            "Region of the Americas",
            "South-East Asia Region",
            "European Region",
            "Eastern Mediterranean Region",
            "Western Pacific Region",
            "WHO region",
        }
    )
    return by_id.fillna(False) | by_name


def process_gbd_primary() -> pd.DataFrame:
    frames = []
    for source_file, frame in csv_frames_from_gbd() or []:
        try:
            candidate = canonicalise_gbd(frame, source_file)
        except ValueError:
            continue
        if not candidate.empty:
            frames.append(candidate)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["location", "year", "rate", "lower", "upper"])
    out = out.query("1990 <= year <= 2023 and rate > 0 and lower > 0 and upper > 0")

    # Separately overlapping exports may repeat the same location-year. Identical repeats are
    # removed above; a genuine value conflict is a data problem and must not be silently averaged.
    conflicts = out[out.duplicated(["location", "year"], keep=False)]
    if not conflicts.empty:
        offenders = (
            conflicts.groupby(["location", "year"])["rate"].nunique().loc[lambda s: s > 1].index.tolist()
        )
        raise ValueError(
            "Conflicting GBD values for the same location-year across exports "
            f"({len(offenders)} pairs, e.g. {offenders[:5]}). Resolve the source files; "
            "the pipeline will not choose between disagreeing exports."
        )

    aggregates = out[is_who_region_aggregate(out)]
    countries = out[~is_who_region_aggregate(out)]
    if not aggregates.empty:
        aggregates.to_csv(PROCESSED / "gbd_who_region_aggregates.csv", index=False)
    countries.to_csv(PROCESSED / "gbd_primary_mortality.csv", index=False)
    return countries


# GBD age bands as exported, mapped to the WHO world standard weights above. The <20 band carries
# the summed 0-4, 5-9, 10-14 and 15-19 weights. Breast cancer incidence below 20 is near zero, so
# the within-band distribution assumption this makes is immaterial for the standardised total.
GBD_AGE_BAND_WEIGHTS = {
    "<20 years": float(WHO_STANDARD_WEIGHTS[0:4].sum()),
    "20-24 years": float(WHO_STANDARD_WEIGHTS[4]),
    "25-29 years": float(WHO_STANDARD_WEIGHTS[5]),
    "30-34 years": float(WHO_STANDARD_WEIGHTS[6]),
    "35-39 years": float(WHO_STANDARD_WEIGHTS[7]),
    "40-44 years": float(WHO_STANDARD_WEIGHTS[8]),
    "45-49 years": float(WHO_STANDARD_WEIGHTS[9]),
    "50-54 years": float(WHO_STANDARD_WEIGHTS[10]),
    "55-59 years": float(WHO_STANDARD_WEIGHTS[11]),
    "60-64 years": float(WHO_STANDARD_WEIGHTS[12]),
    "65-69 years": float(WHO_STANDARD_WEIGHTS[13]),
    "70-74 years": float(WHO_STANDARD_WEIGHTS[14]),
    "75-79 years": float(WHO_STANDARD_WEIGHTS[15]),
    "80-84 years": float(WHO_STANDARD_WEIGHTS[16]),
    "85+ years": float(WHO_STANDARD_WEIGHTS[17]),
}


def constructed_age_standardised_incidence() -> pd.DataFrame:
    """Build an age-standardised female breast-cancer incidence rate for 1990-2023.

    GBD's own age-standardised incidence rows were exported for 2017-2023 only, but the
    age-specific incidence rates cover the full period. Direct standardisation with the WHO world
    standard recovers a comparable series for every year, which the distributed-lag analysis needs.
    The result is not numerically identical to GBD's own standardisation, which uses the GBD
    standard population; `validate_constructed_incidence` quantifies the difference on the years
    where both exist.
    """
    rows = []
    for source_file, frame in csv_frames_from_gbd() or []:
        frame = frame.rename(columns={c: normalise_column(c) for c in frame.columns})
        needed = {"measure_name", "metric_name", "age_name", "sex_name", "location_name", "year", "val"}
        if not needed.issubset(frame.columns):
            continue
        mask = (
            frame["measure_name"].astype(str).str.casefold().eq("incidence")
            & frame["metric_name"].astype(str).str.casefold().eq("rate")
            & frame["sex_name"].astype(str).str.casefold().eq("female")
            & frame["age_name"].isin(GBD_AGE_BAND_WEIGHTS)
        )
        if not mask.any():
            continue
        subset = frame.loc[mask, ["location_name", "location_id", "year", "age_name", "val"]].copy()
        subset["source_file"] = source_file
        rows.append(subset)
    if not rows:
        return pd.DataFrame()

    band = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["location_name", "year", "age_name", "val"]
    )
    band["weight"] = band["age_name"].map(GBD_AGE_BAND_WEIGHTS)
    band["contribution"] = pd.to_numeric(band["val"], errors="coerce") * band["weight"]

    complete = band.groupby(["location_name", "location_id", "year"], dropna=False).agg(
        asir_per_100k=("contribution", "sum"),
        bands=("age_name", "nunique"),
    ).reset_index()
    # Only accept a standardised value where every age band is present; a missing band would
    # silently understate the rate rather than fail.
    complete = complete[complete["bands"] == len(GBD_AGE_BAND_WEIGHTS)].drop(columns=["bands"])
    complete = complete.rename(columns={"location_name": "location"})
    complete = complete[~is_who_region_aggregate(complete)]
    complete.to_csv(PROCESSED / "gbd_constructed_incidence_asr.csv", index=False)
    return complete


def validate_constructed_incidence(constructed: pd.DataFrame) -> pd.DataFrame:
    """Compare the constructed incidence series with GBD's own on the overlapping years."""
    published = []
    for source_file, frame in csv_frames_from_gbd() or []:
        frame = frame.rename(columns={c: normalise_column(c) for c in frame.columns})
        if not {"measure_name", "metric_name", "age_name", "location_name", "year", "val"}.issubset(frame.columns):
            continue
        mask = (
            frame["measure_name"].astype(str).str.casefold().eq("incidence")
            & frame["metric_name"].astype(str).str.casefold().eq("rate")
            & frame["age_name"].astype(str).str.casefold().eq("age-standardized")
        )
        if mask.any():
            published.append(frame.loc[mask, ["location_name", "year", "val"]])
    if not published or constructed.empty:
        return pd.DataFrame()
    pub = pd.concat(published, ignore_index=True).drop_duplicates(["location_name", "year", "val"])
    pub = pub.rename(columns={"location_name": "location", "val": "gbd_asir"})
    merged = constructed.merge(pub, on=["location", "year"], how="inner")
    if merged.empty:
        return merged
    merged["ratio"] = merged["asir_per_100k"] / merged["gbd_asir"]
    merged.to_csv(RESULTS / "constructed_incidence_validation.csv", index=False)
    return merged


# WPP five-year bands collapsed onto the GBD export's bands. GBD supplies <20 and 85+ as single
# bands, so the finer WPP groups either side are summed into them.
WPP_TO_GBD_BAND = {
    **{band: "<20 years" for band in ("0-4", "5-9", "10-14", "15-19")},
    **{f"{start}-{start + 4}": f"{start}-{start + 4} years" for start in range(20, 85, 5)},
    **{band: "85+ years" for band in ("85-89", "90-94", "95-99", "100+")},
}


def process_un_wpp() -> pd.DataFrame:
    """Female population by country, year and GBD age band, from WPP 2024 medium variant.

    Values in the source are thousands of persons and are converted to persons here. Aggregate
    rows (regions, income groups, SDG groupings) carry no ISO3 code and are dropped.
    """
    base = latest_snapshot_dir("un_wpp")
    if base is None:
        return pd.DataFrame()
    path = base / "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz"
    if not path.exists():
        return pd.DataFrame()

    keep = ["ISO3_code", "Time", "AgeGrp", "PopFemale"]
    chunks = []
    for chunk in pd.read_csv(
        path, compression="gzip", usecols=keep, chunksize=500_000, encoding="utf-8-sig", low_memory=False
    ):
        chunk = chunk[chunk["ISO3_code"].astype(str).str.fullmatch(r"[A-Z]{3}")]
        chunk = chunk[chunk["AgeGrp"].isin(WPP_TO_GBD_BAND)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()

    frame = pd.concat(chunks, ignore_index=True)
    frame["age_band"] = frame["AgeGrp"].map(WPP_TO_GBD_BAND)
    frame["female_population"] = pd.to_numeric(frame["PopFemale"], errors="coerce") * 1000.0
    out = (
        frame.groupby(["ISO3_code", "Time", "age_band"], as_index=False)["female_population"].sum()
        .rename(columns={"ISO3_code": "iso3", "Time": "year"})
    )
    out["year"] = out["year"].astype(int)
    out.to_csv(PROCESSED / "wpp_female_population_by_band.csv", index=False)
    return out


# Primary window and coverage rule, matching analyse_primary so the confirmatory model classifies
# exactly the same country set as the diagnostic.
PRIMARY_START = 2015
PRIMARY_END = 2023
MIN_YEARS_PRIMARY = 6


def world_bank_country_groups() -> pd.DataFrame:
    """ISO3 with World Bank region and income group, for partial pooling."""
    base = latest_snapshot_dir("world_bank")
    if base is None:
        return pd.DataFrame()
    path = base / "country_metadata.json"
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    records = [
        {
            "iso3": row["id"],
            "region": (row.get("region") or {}).get("value", "").strip(),
            "income_group": (row.get("incomeLevel") or {}).get("value", "").strip(),
        }
        for row in rows or []
        if (row.get("region") or {}).get("id") != "NA" and row.get("id")
    ]
    return pd.DataFrame(records)


def _country_slope_with_measurement_error(frame: pd.DataFrame) -> tuple[float, float, int] | None:
    """Weighted least squares of log rate on year, weighting by reconstructed measurement variance.

    Returns the slope, its sampling variance, and the number of years used.
    """
    frame = frame.dropna(subset=["rate", "lower", "upper"])
    if len(frame) < MIN_YEARS_PRIMARY:
        return None
    log_rate = np.log(frame["rate"].to_numpy(dtype=float))
    sd = interval_to_log_sd(
        frame["rate"].to_numpy(dtype=float),
        frame["lower"].to_numpy(dtype=float),
        frame["upper"].to_numpy(dtype=float),
    )
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    if np.any(~np.isfinite(sd)):
        return None
    years = frame["year"].to_numpy(dtype=float)
    weights = 1.0 / np.square(sd)
    sum_w = float(weights.sum())
    if sum_w <= 0:
        return None
    # Both year and log rate must be centred on their *weighted* means, otherwise the numerator
    # retains an intercept term and the slope is badly biased.
    year_centred = years - float(np.sum(weights * years) / sum_w)
    log_rate_centred = log_rate - float(np.sum(weights * log_rate) / sum_w)
    sum_wx2 = float(np.sum(weights * np.square(year_centred)))
    if sum_wx2 <= 0:
        return None
    slope = float(np.sum(weights * year_centred * log_rate_centred) / sum_wx2)
    variance = float(1.0 / sum_wx2)
    return slope, variance, len(frame)


def _fit_group_hyperparameters(slopes: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood mean and between-country slope SD for one pooling group."""
    if len(slopes) == 1:
        return float(slopes[0]), 0.0

    def negative_log_likelihood(log_tau: float) -> float:
        tau2 = float(np.exp(log_tau)) ** 2
        total = variances + tau2
        mean = float(np.sum(slopes / total) / np.sum(1.0 / total))
        return float(0.5 * np.sum(np.log(total) + np.square(slopes - mean) / total))

    best_log_tau, best_value = None, np.inf
    for candidate in np.linspace(-12.0, 1.0, 200):
        value = negative_log_likelihood(candidate)
        if value < best_value:
            best_value, best_log_tau = value, candidate
    tau = float(np.exp(best_log_tau)) if best_log_tau is not None else 0.0
    if tau < 1e-5:
        tau = 0.0
    total = variances + tau**2
    mean = float(np.sum(slopes / total) / np.sum(1.0 / total))
    return mean, tau


def _pooling_group_lookup(grouping: str) -> dict[str, str]:
    """ISO3 to pooling-group label for the requested grouping variable."""
    if grouping == "who_region":
        crosswalk = who_region_crosswalk()
        if crosswalk.empty:
            return {}
        return dict(zip(crosswalk["iso3"], crosswalk["who_region"]))
    groups = world_bank_country_groups()
    if groups.empty or grouping not in groups.columns:
        return {}
    return dict(zip(groups["iso3"], groups[grouping]))


def analyse_hierarchical_trajectory(
    frame: pd.DataFrame,
    grouping: str = "region",
    output_name: str = "hierarchical_country_trajectory.csv",
) -> pd.DataFrame:
    """Confirmatory country classification with partial pooling of slopes within groups.

    An empirical-Bayes normal-normal hierarchy, not a Monte Carlo posterior: country slopes are
    estimated by weighted least squares that carries the reconstructed measurement variance, then
    shrunk towards a group mean whose between-country spread is estimated by maximum likelihood.
    Countries with noisy or short series shrink further towards their group.

    The reconstructed intervals cannot recover GBD's cross-year uncertainty correlation, so the
    posterior spread is anti-conservative in the same way the diagnostic is. Draw-level data would
    be required to remove that limitation.
    """
    window = frame[(frame["year"] >= PRIMARY_START) & (frame["year"] <= PRIMARY_END)]
    if window.empty:
        return pd.DataFrame()

    mapping, _ = gbd_locations_to_iso3(window["location"])
    group_lookup = _pooling_group_lookup(grouping)
    unassigned_label = UNASSIGNED_WHO_REGION if grouping == "who_region" else "Unclassified"

    stage_one = []
    for location, subset in window.groupby("location", sort=False):
        estimate = _country_slope_with_measurement_error(subset)
        if estimate is None:
            continue
        slope, variance, years = estimate
        iso3 = mapping.get(location)
        stage_one.append(
            {
                "location": location,
                "iso3": iso3,
                "group": group_lookup.get(iso3) or unassigned_label,
                "slope": slope,
                "variance": variance,
                "n_years": years,
            }
        )
    if not stage_one:
        return pd.DataFrame()

    table = pd.DataFrame(stage_one)
    threshold = math.log(1.0 - 0.025)  # APC of -2.5% on the log scale
    records = []
    for group_name, subset in table.groupby("group", sort=False):
        mean, tau = _fit_group_hyperparameters(
            subset["slope"].to_numpy(dtype=float), subset["variance"].to_numpy(dtype=float)
        )
        for row in subset.itertuples():
            if tau == 0.0:
                posterior_mean, posterior_var = mean, row.variance
                shrinkage = 1.0
            else:
                precision_data = 1.0 / row.variance
                precision_prior = 1.0 / tau**2
                posterior_var = 1.0 / (precision_data + precision_prior)
                posterior_mean = posterior_var * (precision_data * row.slope + precision_prior * mean)
                shrinkage = float(precision_prior / (precision_data + precision_prior))
            posterior_sd = math.sqrt(posterior_var)
            probability = float(norm.cdf((threshold - posterior_mean) / posterior_sd))
            records.append(
                {
                    "location": row.location,
                    "iso3": row.iso3,
                    "group": group_name,
                    "n_years": row.n_years,
                    "apc_unpooled": apc_from_log_slope(row.slope),
                    "apc_pooled": apc_from_log_slope(posterior_mean),
                    "apc_pooled_lower": apc_from_log_slope(posterior_mean - 1.96 * posterior_sd),
                    "apc_pooled_upper": apc_from_log_slope(posterior_mean + 1.96 * posterior_sd),
                    "posterior_sd_log": posterior_sd,
                    "shrinkage_to_group": shrinkage,
                    "group_mean_apc": apc_from_log_slope(mean),
                    "group_tau_log": tau,
                    "p_apc_le_minus_2_5": probability,
                    "trajectory_class": classify_probability(probability),
                    "uncertainty_method": "empirical_bayes_partial_pooling_confirmatory",
                }
            )
    result = pd.DataFrame(records).sort_values("p_apc_le_minus_2_5", ascending=False)
    result.to_csv(RESULTS / output_name, index=False)
    return result


# ---------------------------------------------------------------------------
# WHO region mapping and prespecified subgroup stratification (protocol objective 6)
# ---------------------------------------------------------------------------

WHO_REGION_NAMES = {
    "AFR": "African Region",
    "AMR": "Region of the Americas",
    "SEAR": "South-East Asia Region",
    "EUR": "European Region",
    "EMR": "Eastern Mediterranean Region",
    "WPR": "Western Pacific Region",
}
UNASSIGNED_WHO_REGION = "Not assigned to a WHO region"

# WHO Mortality Database country labels the World Bank name lookup does not resolve. Hong Kong SAR
# and "United Kingdom, Northern Ireland" are deliberately excluded rather than aliased: the first is
# not a separate location in this GBD export and the second is subnational.
WHO_MDB_NAME_ALIASES = {"Czech Republic": "CZE", "Turkey": "TUR"}


def who_region_crosswalk() -> pd.DataFrame:
    """ISO3 to WHO region, read from the ParentLocationCode carried by every GHO country row.

    This is WHO's own regional assignment as served alongside the indicator values, so no
    hand-entered membership list is required and the mapping cannot drift from the source. Every
    downloaded GHO snapshot is read; a country assigned to two different regions by two indicators
    is recorded as a conflict rather than silently resolved.
    """
    base = RAW / "who_gho"
    columns = ["iso3", "who_region_code", "who_region"]
    if not base.exists():
        return pd.DataFrame(columns=columns)
    assignments: dict[str, dict[str, str]] = {}
    conflicts: list[dict[str, str]] = []
    for path in sorted(base.rglob("page_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in payload.get("value", []) or []:
            if row.get("SpatialDimType") != "COUNTRY":
                continue
            iso3, code = row.get("SpatialDim"), row.get("ParentLocationCode")
            if not iso3 or not code:
                continue
            previous = assignments.get(iso3)
            if previous is not None and previous["who_region_code"] != code:
                conflicts.append(
                    {"iso3": iso3, "first_seen": previous["who_region_code"], "also_seen": code}
                )
                continue
            assignments[iso3] = {
                "iso3": iso3,
                "who_region_code": code,
                "who_region": WHO_REGION_NAMES.get(code, row.get("ParentLocation") or code),
            }
    if conflicts:
        pd.DataFrame(conflicts).drop_duplicates().to_csv(
            METADATA / "who_region_conflicts.csv", index=False
        )
    frame = pd.DataFrame(sorted(assignments.values(), key=lambda record: record["iso3"]), columns=columns)
    if not frame.empty:
        frame.to_csv(METADATA / "who_region_crosswalk.csv", index=False)
    return frame


def registry_series_availability() -> pd.DataFrame:
    """Years of usable observed breast-cancer ASDR per country in the WHO Mortality Database.

    Used as the data-quality axis that reflects national vital registration rather than the
    precision of the modelled outcome.
    """
    path = PROCESSED / "who_mdb_breast_cancer_observed.csv"
    columns = ["iso3", "registry_years_in_window"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    mdb = pd.read_csv(path).dropna(subset=["asdr_per_100k"])
    mdb = mdb[(mdb["year"] >= PRIMARY_START) & (mdb["year"] <= PRIMARY_END)]
    if mdb.empty:
        return pd.DataFrame(columns=columns)
    mapping, _ = gbd_locations_to_iso3(mdb["country"])
    mapping = {**mapping, **WHO_MDB_NAME_ALIASES}
    mdb = mdb.assign(iso3=mdb["country"].map(mapping)).dropna(subset=["iso3"])
    counts = mdb.groupby("iso3", as_index=False)["year"].nunique()
    return counts.rename(columns={"year": "registry_years_in_window"})


def baseline_and_precision(frame: pd.DataFrame) -> pd.DataFrame:
    """Baseline mortality and outcome precision per location over the primary window.

    Precision is the median relative width of the GBD 95% uncertainty interval, that is
    (upper - lower) / rate. It measures how well the modelled outcome itself is pinned down, which
    is available for every location, unlike registry coverage.
    """
    window = frame[(frame["year"] >= PRIMARY_START) & (frame["year"] <= PRIMARY_END)].copy()
    if window.empty:
        return pd.DataFrame(columns=["location", "baseline_asmr_2015", "median_relative_ui_width"])
    window["relative_ui_width"] = (window["upper"] - window["lower"]) / window["rate"]
    baseline = (
        window[window["year"] == PRIMARY_START]
        .groupby("location", as_index=False)["rate"]
        .first()
        .rename(columns={"rate": "baseline_asmr_2015"})
    )
    precision = (
        window.groupby("location", as_index=False)["relative_ui_width"]
        .median()
        .rename(columns={"relative_ui_width": "median_relative_ui_width"})
    )
    return baseline.merge(precision, on="location", how="outer")


def _tertile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    """Tertiles by rank, so ties and skew cannot leave a stratum empty."""
    usable = values.dropna()
    if usable.nunique() < 3:
        return pd.Series(pd.NA, index=values.index, dtype="object")
    ranked = values.rank(method="first")
    cut = pd.qcut(ranked, 3, labels=labels)
    return cut.astype("object").where(values.notna(), other=pd.NA)


NO_SURVIVAL_ESTIMATE = "No WHO survival estimate"

SURVIVAL_SOURCE_CLASS_LABELS = [
    "Most registry-informed third",
    "Middle third",
    "Least registry-informed third",
]


def who_survival_source_class() -> pd.DataFrame:
    """Source class of each country's WHO 5-year net survival estimate, per iso3.

    WHO publishes no per-country flag separating survival estimates taken from a registry or a
    Member State submission from those predicted by its Bayesian hierarchical model, so the class
    is read off the width of the published uncertainty interval, which widens as country-specific
    data thin out. `04_code/audit_survival_circularity.py` validates that reading against years of
    observed registry series and confirms that the width does not track the mortality level.

    The survival VALUE is deliberately not returned. That audit rejects it as an explanatory
    variable, because it shares most of its variance with the mortality-to-incidence ratio derived
    from this study's own outcome.
    """
    path = PROCESSED / "who_survival.csv"
    columns = ["iso3", "survival_ui_width"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    survival = pd.read_csv(path)
    if "SpatialDimType" not in survival.columns:
        return pd.DataFrame(columns=columns)
    survival = survival[survival["SpatialDimType"].eq("COUNTRY")]
    needed = {"SpatialDim", "Low", "High"}
    if survival.empty or not needed.issubset(survival.columns):
        return pd.DataFrame(columns=columns)
    width = pd.to_numeric(survival["High"], errors="coerce") - pd.to_numeric(
        survival["Low"], errors="coerce"
    )
    frame = pd.DataFrame({"iso3": survival["SpatialDim"], "survival_ui_width": width})
    return frame.dropna(subset=["survival_ui_width"]).drop_duplicates("iso3")


SUBGROUP_AXES = {
    "who_region": "WHO region",
    "income_group": "World Bank income group",
    "world_bank_region": "World Bank region",
    "baseline_mortality_tertile": "Baseline 2015 age-standardised mortality rate",
    "outcome_precision_tertile": "Median relative width of the GBD 95% uncertainty interval",
    "registry_series": "Usable WHO Mortality Database registry series, 2015-2023",
    "survival_source_class": "Source class of the WHO 5-year net survival estimate",
}


def build_subgroup_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per analysed country carrying its trend estimate and every stratification variable."""
    window = frame[(frame["year"] >= PRIMARY_START) & (frame["year"] <= PRIMARY_END)]
    if window.empty:
        return pd.DataFrame()
    rows = []
    for location, subset in window.groupby("location", sort=True):
        estimate = _country_slope_with_measurement_error(subset)
        if estimate is None:
            continue
        slope, variance, years = estimate
        rows.append({"location": location, "slope": slope, "variance": variance, "n_years": years})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)

    mapping, _ = gbd_locations_to_iso3(table["location"])
    table["iso3"] = table["location"].map(mapping)

    who = who_region_crosswalk()
    if not who.empty:
        table = table.merge(who, on="iso3", how="left")
    else:
        table["who_region_code"] = pd.NA
        table["who_region"] = pd.NA
    table["who_region"] = table["who_region"].fillna(UNASSIGNED_WHO_REGION)

    banks = world_bank_country_groups()
    if not banks.empty:
        table = table.merge(
            banks.rename(columns={"region": "world_bank_region"}), on="iso3", how="left"
        )
    else:
        table["world_bank_region"] = pd.NA
        table["income_group"] = pd.NA
    for column in ("world_bank_region", "income_group"):
        table[column] = table[column].replace("", pd.NA).fillna("Unclassified")

    table = table.merge(baseline_and_precision(frame), on="location", how="left")
    table["baseline_mortality_tertile"] = _tertile_labels(
        table["baseline_asmr_2015"], ["Lowest third", "Middle third", "Highest third"]
    )
    table["outcome_precision_tertile"] = _tertile_labels(
        table["median_relative_ui_width"], ["Narrowest third", "Middle third", "Widest third"]
    )

    registry = registry_series_availability()
    if not registry.empty:
        table = table.merge(registry, on="iso3", how="left")
    else:
        table["registry_years_in_window"] = pd.NA
    years_available = pd.to_numeric(table["registry_years_in_window"], errors="coerce").fillna(0)
    table["registry_years_in_window"] = years_available.astype(int)
    table["registry_series"] = np.where(
        years_available >= MIN_YEARS_PRIMARY,
        f"At least {MIN_YEARS_PRIMARY} registry years",
        np.where(years_available > 0, "1 to 5 registry years", "No registry series"),
    )

    survival = who_survival_source_class()
    if not survival.empty:
        table = table.merge(survival, on="iso3", how="left")
    else:
        table["survival_ui_width"] = np.nan
    table["survival_source_class"] = _tertile_labels(
        table["survival_ui_width"], SURVIVAL_SOURCE_CLASS_LABELS
    )
    table["survival_source_class"] = table["survival_source_class"].fillna(NO_SURVIVAL_ESTIMATE)

    table["apc"] = apc_from_log_slope(table["slope"])
    table.to_csv(RESULTS / "subgroup_country_assignments.csv", index=False)
    return table


def _random_effects_summary(slopes: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """Inverse-variance random-effects summary of country log slopes within one stratum.

    The stratum mean and between-country spread are fitted by the same maximum-likelihood routine
    the confirmatory model uses, so a stratum summary and a pooling group are the same object.
    Heterogeneity is summarised by Cochran's Q about the fixed-effect mean and by I squared.
    """
    count = len(slopes)
    mean, tau = _fit_group_hyperparameters(slopes, variances)
    total = variances + tau**2
    standard_error = float(1.0 / math.sqrt(float(np.sum(1.0 / total))))
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * slopes) / np.sum(fixed_weights))
    q_statistic = float(np.sum(fixed_weights * np.square(slopes - fixed_mean)))
    degrees = count - 1
    if degrees > 0 and q_statistic > degrees:
        i_squared = 100.0 * (q_statistic - degrees) / q_statistic
    else:
        i_squared = 0.0
    return {
        "countries": count,
        "mean_log_slope": mean,
        "standard_error_log": standard_error,
        "apc_mean": apc_from_log_slope(mean),
        "apc_lower": apc_from_log_slope(mean - 1.96 * standard_error),
        "apc_upper": apc_from_log_slope(mean + 1.96 * standard_error),
        "between_country_sd_log": tau,
        "cochran_q": q_statistic,
        "q_degrees_of_freedom": degrees,
        "i_squared_percent": i_squared,
    }


def analyse_subgroups(
    frame: pd.DataFrame, classifications: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Stratified trend summaries and between-stratum heterogeneity tests.

    Country slopes enter unpooled, so a stratum summary is not shrunk twice. Classification counts
    are carried across from the confirmatory partially pooled model when it is available.
    """
    table = build_subgroup_frame(frame)
    if table.empty:
        return {}
    if classifications is not None and not classifications.empty:
        classes = classifications[["location", "trajectory_class"]].drop_duplicates("location")
        table = table.merge(classes, on="location", how="left")
    else:
        table["trajectory_class"] = pd.NA

    overall = _random_effects_summary(
        table["slope"].to_numpy(dtype=float), table["variance"].to_numpy(dtype=float)
    )
    summary_rows = [{"axis": "overall", "axis_label": "All analysed countries", "level": "All countries", **overall}]
    heterogeneity_rows = []

    for axis, axis_label in SUBGROUP_AXES.items():
        if axis not in table.columns:
            continue
        levels = table.dropna(subset=[axis])
        if levels.empty:
            continue
        axis_rows = []
        for level, subset in levels.groupby(axis, sort=True):
            stratum = _random_effects_summary(
                subset["slope"].to_numpy(dtype=float), subset["variance"].to_numpy(dtype=float)
            )
            counts = subset["trajectory_class"].value_counts()
            stratum.update(
                {
                    "axis": axis,
                    "axis_label": axis_label,
                    "level": str(level),
                    "on_trajectory": int(counts.get("on_trajectory", 0)),
                    "uncertain": int(counts.get("uncertain", 0)),
                    "off_trajectory": int(counts.get("off_trajectory", 0)),
                    "median_apc": float(subset["apc"].median()),
                    "apc_first_quartile": float(subset["apc"].quantile(0.25)),
                    "apc_third_quartile": float(subset["apc"].quantile(0.75)),
                }
            )
            axis_rows.append(stratum)
        summary_rows.extend(axis_rows)
        heterogeneity_rows.append(_between_stratum_test(axis, axis_label, axis_rows))

    summary = pd.DataFrame(summary_rows)
    ordered = [
        "axis",
        "axis_label",
        "level",
        "countries",
        "apc_mean",
        "apc_lower",
        "apc_upper",
        "median_apc",
        "apc_first_quartile",
        "apc_third_quartile",
        "on_trajectory",
        "uncertain",
        "off_trajectory",
        "between_country_sd_log",
        "cochran_q",
        "q_degrees_of_freedom",
        "i_squared_percent",
        "mean_log_slope",
        "standard_error_log",
    ]
    summary = summary[[column for column in ordered if column in summary.columns]]
    summary.to_csv(RESULTS / "subgroup_stratification.csv", index=False)
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    if not heterogeneity.empty:
        heterogeneity.to_csv(RESULTS / "subgroup_heterogeneity.csv", index=False)
    return {"assignments": table, "summary": summary, "heterogeneity": heterogeneity}


def _between_stratum_test(axis: str, axis_label: str, rows: list[dict[str, float]]) -> dict[str, float]:
    """Cochran's Q across stratum means, weighted by the inverse of their random-effects variances."""
    means = np.array([row["mean_log_slope"] for row in rows], dtype=float)
    errors = np.array([row["standard_error_log"] for row in rows], dtype=float)
    usable = np.isfinite(means) & np.isfinite(errors) & (errors > 0)
    means, errors = means[usable], errors[usable]
    degrees = len(means) - 1
    if degrees < 1:
        return {
            "axis": axis,
            "axis_label": axis_label,
            "levels": int(len(means)),
            "q_between": float("nan"),
            "degrees_of_freedom": degrees,
            "p_value": float("nan"),
        }
    weights = 1.0 / np.square(errors)
    grand_mean = float(np.sum(weights * means) / np.sum(weights))
    q_between = float(np.sum(weights * np.square(means - grand_mean)))
    return {
        "axis": axis,
        "axis_label": axis_label,
        "levels": int(len(means)),
        "q_between": q_between,
        "degrees_of_freedom": degrees,
        "p_value": float(chi2.sf(q_between, degrees)),
    }


def validate_who_region_aggregates(subgroups: pd.DataFrame) -> pd.DataFrame:
    """Compare the country-level WHO regional mean against GBD's own regional aggregate series.

    The two quantities are not identical by construction. GBD's aggregate is a population-weighted
    regional rate, whereas the stratum mean weights countries by the precision of their own trend,
    so agreement is a coherence check and disagreement is informative about weighting rather than
    evidence of an error in either series.
    """
    path = PROCESSED / "gbd_who_region_aggregates.csv"
    if not path.exists() or subgroups.empty:
        return pd.DataFrame()
    aggregates = pd.read_csv(path)
    aggregates = aggregates[
        (aggregates["year"] >= PRIMARY_START) & (aggregates["year"] <= PRIMARY_END)
    ]
    if aggregates.empty:
        return pd.DataFrame()
    rows = []
    for location, subset in aggregates.groupby("location", sort=True):
        estimate = _country_slope_with_measurement_error(subset)
        if estimate is None:
            continue
        slope, variance, years = estimate
        matched = subgroups[subgroups["who_region"] == location]
        if matched.empty:
            country_summary = None
        else:
            country_summary = _random_effects_summary(
                matched["slope"].to_numpy(dtype=float), matched["variance"].to_numpy(dtype=float)
            )
        rows.append(
            {
                "who_region": location,
                "aggregate_years": years,
                "aggregate_apc": apc_from_log_slope(slope),
                "aggregate_apc_lower": apc_from_log_slope(slope - 1.96 * math.sqrt(variance)),
                "aggregate_apc_upper": apc_from_log_slope(slope + 1.96 * math.sqrt(variance)),
                "countries_in_region": country_summary["countries"] if country_summary else 0,
                "country_mean_apc": country_summary["apc_mean"] if country_summary else float("nan"),
                "country_mean_apc_lower": country_summary["apc_lower"] if country_summary else float("nan"),
                "country_mean_apc_upper": country_summary["apc_upper"] if country_summary else float("nan"),
                "difference_apc_points": (
                    apc_from_log_slope(slope) - country_summary["apc_mean"] if country_summary else float("nan")
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(RESULTS / "who_region_aggregate_validation.csv", index=False)
    return result


def gbd_age_specific_death_rates() -> pd.DataFrame:
    """Female breast-cancer death rate per 100,000 by country, year and GBD age band."""
    rows = []
    for source_file, frame in csv_frames_from_gbd() or []:
        frame = frame.rename(columns={c: normalise_column(c) for c in frame.columns})
        needed = {"measure_name", "metric_name", "age_name", "sex_name", "location_name", "year", "val"}
        if not needed.issubset(frame.columns):
            continue
        mask = (
            frame["measure_name"].astype(str).str.casefold().eq("deaths")
            & frame["metric_name"].astype(str).str.casefold().eq("rate")
            & frame["sex_name"].astype(str).str.casefold().eq("female")
            & frame["age_name"].isin(GBD_AGE_BAND_WEIGHTS)
        )
        if mask.any():
            keep = ["location_name", "location_id", "year", "age_name", "val"]
            rows.append(frame.loc[mask, [c for c in keep if c in frame.columns]])
    if not rows:
        return pd.DataFrame()
    band = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["location_name", "year", "age_name", "val"]
    )
    band = band.rename(columns={"location_name": "location", "age_name": "age_band", "val": "rate_per_100k"})
    band = band[~is_who_region_aggregate(band)]
    band["rate_per_100k"] = pd.to_numeric(band["rate_per_100k"], errors="coerce")
    band = band.dropna(subset=["rate_per_100k"])
    band.to_csv(PROCESSED / "gbd_age_specific_death_rates.csv", index=False)
    return band


FORECAST_END = 2040
BACKTEST_TRAIN_END = 2019  # fit on 2015-2019, score the held-out 2020-2023


def _log_linear_forecast(years: np.ndarray, log_rates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Ordinary least squares on log rate, evaluated at the target years."""
    centred = years - years.mean()
    denominator = float(np.sum(np.square(centred)))
    if denominator <= 0:
        return np.full(len(targets), np.nan)
    slope = float(np.sum(centred * (log_rates - log_rates.mean())) / denominator)
    intercept = float(log_rates.mean() - slope * years.mean())
    return np.exp(intercept + slope * targets)


def backtest_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold out 2020-2023, fit on 2015-2019, and score the extrapolation.

    The protocol requires forecasts to pass a historical back-test before being reported, so this
    runs before any projection is published rather than as an afterthought.
    """
    window = frame[(frame["year"] >= PRIMARY_START) & (frame["year"] <= PRIMARY_END)]
    rows = []
    for location, subset in window.groupby("location", sort=True):
        train = subset[subset["year"] <= BACKTEST_TRAIN_END]
        test = subset[subset["year"] > BACKTEST_TRAIN_END]
        if len(train) < 4 or test.empty:
            continue
        predicted = _log_linear_forecast(
            train["year"].to_numpy(dtype=float),
            np.log(train["rate"].to_numpy(dtype=float)),
            test["year"].to_numpy(dtype=float),
        )
        actual = test["rate"].to_numpy(dtype=float)
        if not np.all(np.isfinite(predicted)):
            continue
        error = (predicted - actual) / actual
        rows.append(
            {
                "location": location,
                "train_years": len(train),
                "test_years": len(test),
                "mean_absolute_percentage_error": float(100.0 * np.mean(np.abs(error))),
                "mean_percentage_error": float(100.0 * np.mean(error)),
                "max_absolute_percentage_error": float(100.0 * np.max(np.abs(error))),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(RESULTS / "forecast_backtest.csv", index=False)
    return result


def forecast_scenarios(frame: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Project age-standardised mortality and deaths to 2040 under three explicit scenarios.

    Scenarios, all applied to the observed 2023 rate:
      * continuation  - each country keeps its own fitted 2015-2023 annual percent change;
      * benchmark     - every country achieves the GBCI -2.5% per year;
      * stagnation    - rates hold at their 2023 value.

    Deaths are generated with WPP medium-variant female population so that population growth and
    ageing are not confused with changes in the rate. Age-standardised rates are applied to total
    female population, which yields a comparable standardised count, not a crude death count.
    """
    window = frame[(frame["year"] >= PRIMARY_START) & (frame["year"] <= PRIMARY_END)]
    age_rates = gbd_age_specific_death_rates()
    if window.empty or population.empty or age_rates.empty:
        return pd.DataFrame()

    mapping, _ = gbd_locations_to_iso3(window["location"])
    horizon = np.arange(PRIMARY_END, FORECAST_END + 1)

    # Country trend on the age-standardised rate, which is what the scenarios are defined on.
    slopes = {}
    for location, subset in window.groupby("location", sort=True):
        if subset["year"].nunique() < MIN_YEARS_PRIMARY:
            continue
        years = subset["year"].to_numpy(dtype=float)
        log_rates = np.log(subset["rate"].to_numpy(dtype=float))
        centred = years - years.mean()
        slopes[location] = float(
            np.sum(centred * (log_rates - log_rates.mean())) / np.sum(np.square(centred))
        )

    base = age_rates[age_rates["year"].eq(PRIMARY_END)][["location", "age_band", "rate_per_100k"]]
    base = base[base["location"].isin(slopes)].copy()
    base["iso3"] = base["location"].map(mapping)
    base = base.dropna(subset=["iso3"])

    pop = population[population["year"].between(PRIMARY_END, FORECAST_END)]
    grid = base.merge(pop, on=["iso3", "age_band"], how="inner")
    if grid.empty:
        return pd.DataFrame()
    grid["slope"] = grid["location"].map(slopes)
    grid["elapsed"] = grid["year"] - PRIMARY_END

    scenarios = {
        "continuation": None,  # each country keeps its own fitted trend
        "benchmark_2_5pc": math.log(1.0 - 0.025),
        "stagnation": 0.0,
    }
    pieces = []
    for scenario, fixed_slope in scenarios.items():
        piece = grid.copy()
        piece["scenario"] = scenario
        applied = piece["slope"] if fixed_slope is None else fixed_slope
        # The scenario multiplier is applied proportionally across age bands: a stated assumption,
        # not a claim that every age band moves identically in reality.
        piece["rate_per_100k"] = piece["rate_per_100k"] * np.exp(applied * piece["elapsed"])
        piece["deaths"] = piece["rate_per_100k"] * piece["female_population"] / 100_000.0
        pieces.append(piece)

    detail = pd.concat(pieces, ignore_index=True)
    result = detail.groupby(["location", "iso3", "scenario", "year"], as_index=False).agg(
        deaths=("deaths", "sum"), female_population=("female_population", "sum")
    )
    result.to_csv(RESULTS / "forecast_scenarios_country_year.csv", index=False)

    summary = (
        result[result["year"].eq(FORECAST_END)]
        .groupby("scenario", as_index=False)
        .agg(countries=("location", "nunique"), deaths_2040=("deaths", "sum"))
    )
    baseline = summary.loc[summary["scenario"].eq("stagnation"), "deaths_2040"]
    if not baseline.empty:
        summary["difference_vs_stagnation"] = summary["deaths_2040"] - float(baseline.iloc[0])
    summary["note"] = (
        "Deaths from age-specific rates applied to WPP medium-variant female population. "
        "A scenario contrast under a stated proportional assumption, not a prediction and not an "
        "avoidable-death estimate, which would require incident diagnosis cohorts."
    )
    summary.to_csv(RESULTS / "forecast_scenario_summary.csv", index=False)
    return result


def _logarithmic_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Logarithmic mean L(a,b), equal to a when a == b. Used for an exact LMDI decomposition."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    out = np.zeros_like(first)
    close = np.isclose(first, second)
    out[close] = first[close]
    valid = ~close & (first > 0) & (second > 0)
    out[valid] = (first[valid] - second[valid]) / (np.log(first[valid]) - np.log(second[valid]))
    return out


def decompose_death_change(
    start_year: int = PRIMARY_START, end_year: int = PRIMARY_END
) -> pd.DataFrame:
    """Attribute the change in breast-cancer deaths to four multiplicative factors.

    Deaths in an age band are population x age share x incidence rate x mortality-to-incidence
    ratio. An LMDI decomposition splits the total change exactly into population size, age
    structure, incidence rate, and the post-diagnosis ratio, with no unexplained residual.

    The fourth factor is a crude mortality-to-incidence ratio. It is not survival and must not be
    described as such; it is only a summary of deaths relative to diagnosed cases.
    """
    deaths_rate = gbd_age_specific_death_rates()
    population = process_un_wpp()
    if deaths_rate.empty or population.empty:
        return pd.DataFrame()

    incidence_rows = []
    for source_file, frame in csv_frames_from_gbd() or []:
        frame = frame.rename(columns={c: normalise_column(c) for c in frame.columns})
        if not {"measure_name", "metric_name", "age_name", "sex_name", "location_name", "year", "val"}.issubset(frame.columns):
            continue
        mask = (
            frame["measure_name"].astype(str).str.casefold().eq("incidence")
            & frame["metric_name"].astype(str).str.casefold().eq("rate")
            & frame["sex_name"].astype(str).str.casefold().eq("female")
            & frame["age_name"].isin(GBD_AGE_BAND_WEIGHTS)
        )
        if mask.any():
            incidence_rows.append(frame.loc[mask, ["location_name", "year", "age_name", "val"]])
    if not incidence_rows:
        return pd.DataFrame()
    incidence = pd.concat(incidence_rows, ignore_index=True).drop_duplicates(
        ["location_name", "year", "age_name", "val"]
    )
    incidence = incidence.rename(
        columns={"location_name": "location", "age_name": "age_band", "val": "incidence_per_100k"}
    )

    mapping, _ = gbd_locations_to_iso3(deaths_rate["location"])
    frames = []
    for year in (start_year, end_year):
        piece = deaths_rate[deaths_rate["year"].eq(year)][["location", "age_band", "rate_per_100k"]]
        piece = piece.merge(
            incidence[incidence["year"].eq(year)][["location", "age_band", "incidence_per_100k"]],
            on=["location", "age_band"],
            how="inner",
        )
        piece["iso3"] = piece["location"].map(mapping)
        piece = piece.dropna(subset=["iso3"])
        piece = piece.merge(
            population[population["year"].eq(year)][["iso3", "age_band", "female_population"]],
            on=["iso3", "age_band"],
            how="inner",
        )
        piece["year"] = year
        frames.append(piece)
    if len(frames) != 2:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["incidence_per_100k"] > 0) & (combined["female_population"] > 0)]
    combined["deaths"] = combined["rate_per_100k"] * combined["female_population"] / 100_000.0
    combined["mi_ratio"] = combined["rate_per_100k"] / combined["incidence_per_100k"]
    totals = combined.groupby(["iso3", "year"])["female_population"].transform("sum")
    combined["age_share"] = combined["female_population"] / totals
    combined["total_population"] = totals

    start = combined[combined["year"].eq(start_year)].set_index(["iso3", "age_band"])
    end = combined[combined["year"].eq(end_year)].set_index(["iso3", "age_band"])
    shared = start.index.intersection(end.index)
    start, end = start.loc[shared], end.loc[shared]

    weight = _logarithmic_mean(end["deaths"].to_numpy(), start["deaths"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        effects = pd.DataFrame(
            {
                "iso3": [index[0] for index in shared],
                "age_band": [index[1] for index in shared],
                "deaths_start": start["deaths"].to_numpy(),
                "deaths_end": end["deaths"].to_numpy(),
                "population_effect": weight
                * np.log(end["total_population"].to_numpy() / start["total_population"].to_numpy()),
                "age_structure_effect": weight
                * np.log(end["age_share"].to_numpy() / start["age_share"].to_numpy()),
                "incidence_effect": weight
                * np.log(end["incidence_per_100k"].to_numpy() / start["incidence_per_100k"].to_numpy()),
                "mortality_incidence_ratio_effect": weight
                * np.log(end["mi_ratio"].to_numpy() / start["mi_ratio"].to_numpy()),
            }
        )
    effects = effects.replace([np.inf, -np.inf], np.nan).dropna()
    effects.to_csv(RESULTS / "decomposition_by_country_age.csv", index=False)

    component_columns = [
        "population_effect",
        "age_structure_effect",
        "incidence_effect",
        "mortality_incidence_ratio_effect",
    ]
    summary = pd.DataFrame(
        [
            {
                "start_year": start_year,
                "end_year": end_year,
                "countries": effects["iso3"].nunique(),
                "deaths_start": float(effects["deaths_start"].sum()),
                "deaths_end": float(effects["deaths_end"].sum()),
                "observed_change": float(effects["deaths_end"].sum() - effects["deaths_start"].sum()),
                **{column: float(effects[column].sum()) for column in component_columns},
                "sum_of_components": float(effects[component_columns].sum().sum()),
                "note": (
                    "LMDI decomposition; the fourth factor is a crude mortality-to-incidence ratio "
                    "and is not survival"
                ),
            }
        ]
    )
    summary["residual"] = summary["observed_change"] - summary["sum_of_components"]
    summary.to_csv(RESULTS / "decomposition_summary.csv", index=False)
    return summary


OOP_LAGS = (3, 4, 5, 6, 7)  # pre-specified in the execution brief
INCIDENCE_LAG = 5  # midpoint of the pre-specified lag window


def build_health_system_panel(mortality: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the country-year panel for the health-system association analysis."""
    incidence = constructed_age_standardised_incidence()
    if mortality.empty or incidence.empty:
        return pd.DataFrame(), []

    mapping, unresolved = gbd_locations_to_iso3(mortality["location"])
    panel = mortality.assign(iso3=mortality["location"].map(mapping)).dropna(subset=["iso3"])
    panel = panel[["iso3", "location", "year", "rate"]].rename(columns={"rate": "asmr_per_100k"})

    inc = incidence.assign(iso3=incidence["location"].map(mapping)).dropna(subset=["iso3"])
    panel = panel.merge(inc[["iso3", "year", "asir_per_100k"]], on=["iso3", "year"], how="left")

    wide_path = PROCESSED / "world_bank_indicators_wide.csv"
    if wide_path.exists():
        wb = pd.read_csv(wide_path)
        keep = [c for c in ("iso3", "year", "oop_share_che", "gdp_per_capita_ppp_constant") if c in wb.columns]
        panel = panel.merge(wb[keep].dropna(subset=["iso3"]), on=["iso3", "year"], how="left")

    uhc_path = PROCESSED / "who_uhc_index.csv"
    if uhc_path.exists():
        uhc = pd.read_csv(uhc_path)
        if {"SpatialDim", "TimeDim", "NumericValue", "SpatialDimType"}.issubset(uhc.columns):
            uhc = uhc[uhc["SpatialDimType"] == "COUNTRY"][["SpatialDim", "TimeDim", "NumericValue"]]
            uhc.columns = ["iso3", "year", "uhc_index"]
            uhc = uhc.dropna().drop_duplicates(["iso3", "year"])
            panel = panel.merge(uhc, on=["iso3", "year"], how="left")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel["log_asmr"] = np.log(panel["asmr_per_100k"])
    panel["log_asir"] = np.log(panel["asir_per_100k"])
    panel["log_gdp"] = np.log(panel["gdp_per_capita_ppp_constant"])

    grouped = panel.groupby("iso3", sort=False)
    for lag in OOP_LAGS:
        panel[f"oop_lag{lag}"] = grouped["oop_share_che"].shift(lag)
    panel[f"log_asir_lag{INCIDENCE_LAG}"] = grouped["log_asir"].shift(INCIDENCE_LAG)
    panel.to_csv(PROCESSED / "health_system_panel.csv", index=False)
    return panel, unresolved


def analyse_health_system_lags(mortality: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Within-country distributed-lag association of out-of-pocket share with mortality.

    Log age-standardised mortality on country and year fixed effects, out-of-pocket share at the
    pre-specified 3-7 year lags, lagged incidence, and log GDP. Standard errors are clustered by
    country. This is an association under fixed effects and is not a causal effect of any
    programme; it cannot support a claim about GBCI.
    """
    panel, unresolved = build_health_system_panel(mortality)
    if panel.empty:
        return {}

    lag_terms = [f"oop_lag{lag}" for lag in OOP_LAGS]
    needed = ["log_asmr", f"log_asir_lag{INCIDENCE_LAG}", "log_gdp", *lag_terms]
    model_frame = panel.dropna(subset=needed).copy()
    # Fixed effects need within-country variation over at least two years to be identified.
    counts = model_frame.groupby("iso3")["year"].transform("size")
    model_frame = model_frame[counts >= 2]
    if model_frame.empty or model_frame["iso3"].nunique() < 20:
        return {}

    design = pd.get_dummies(model_frame[["iso3", "year"]].astype({"year": str}), drop_first=True, dtype=float)
    exog = pd.concat([model_frame[[f"log_asir_lag{INCIDENCE_LAG}", "log_gdp", *lag_terms]], design], axis=1)
    exog = sm.add_constant(exog, has_constant="add")
    fitted = sm.OLS(model_frame["log_asmr"], exog).fit(
        cov_type="cluster", cov_kwds={"groups": model_frame["iso3"]}
    )

    rows = []
    for term in [*lag_terms, f"log_asir_lag{INCIDENCE_LAG}", "log_gdp"]:
        rows.append(
            {
                "term": term,
                "coefficient": float(fitted.params[term]),
                "std_error": float(fitted.bse[term]),
                "p_value": float(fitted.pvalues[term]),
                "ci_low": float(fitted.conf_int().loc[term, 0]),
                "ci_high": float(fitted.conf_int().loc[term, 1]),
            }
        )
    coefficients = pd.DataFrame(rows)

    # Cumulative effect of a sustained one-percentage-point higher out-of-pocket share.
    contrast = np.zeros(len(exog.columns))
    for term in lag_terms:
        contrast[list(exog.columns).index(term)] = 1.0
    cumulative = fitted.t_test(contrast)
    summary = pd.DataFrame(
        [
            {
                "quantity": "cumulative_oop_lag3_to_7",
                "coefficient": float(np.squeeze(cumulative.effect)),
                "std_error": float(np.squeeze(cumulative.sd)),
                "p_value": float(np.squeeze(cumulative.pvalue)),
                "percent_change_per_1pp": float(100.0 * np.expm1(np.squeeze(cumulative.effect))),
                "countries": int(model_frame["iso3"].nunique()),
                "observations": int(len(model_frame)),
                "year_min": int(model_frame["year"].min()),
                "year_max": int(model_frame["year"].max()),
                "uhc_countries_with_any_observation": int(panel["uhc_index"].notna().groupby(panel["iso3"]).any().sum())
                if "uhc_index" in panel
                else 0,
                "unresolved_location_names": len(unresolved),
                "interpretation": "within-country association under two-way fixed effects; not a causal effect",
            }
        ]
    )

    coefficients.to_csv(RESULTS / "health_system_lag_coefficients.csv", index=False)
    summary.to_csv(RESULTS / "health_system_lag_summary.csv", index=False)
    return {"coefficients": coefficients, "summary": summary, "panel": model_frame}


def validate_primary(frame: pd.DataFrame) -> list[str]:
    problems = []
    required = {"location", "year", "rate", "lower", "upper"}
    missing = required.difference(frame.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if frame.duplicated(["location", "year"]).any():
        problems.append("duplicate location-year rows")
    if ((frame["lower"] <= 0) | (frame["rate"] <= 0) | (frame["upper"] <= 0)).any():
        problems.append("non-positive rate or interval")
    if ((frame["lower"] > frame["rate"]) | (frame["upper"] < frame["rate"])).any():
        problems.append("uncertainty bounds do not bracket point estimate")
    return problems


def coverage_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame([{"dataset": dataset, "rows": 0, "locations": 0, "min_year": None, "max_year": None}])
    location_col = next((column for column in ("location", "SpatialDim", "iso3", "country") if column in frame.columns), None)
    year_col = next((column for column in ("year", "TimeDim") if column in frame.columns), None)
    return pd.DataFrame(
        [{
            "dataset": dataset,
            "rows": len(frame),
            "locations": frame[location_col].nunique(dropna=True) if location_col else None,
            "min_year": frame[year_col].min() if year_col else None,
            "max_year": frame[year_col].max() if year_col else None,
        }]
    )


def apc_from_log_slope(slope: np.ndarray | float) -> np.ndarray | float:
    return 100.0 * np.expm1(slope)


def classify_probability(probability: float) -> str:
    if probability >= 0.80:
        return "on_trajectory"
    if probability < 0.20:
        return "off_trajectory"
    return "uncertain"


def interval_to_log_sd(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(point <= 0) or np.any(lower <= 0) or np.any(upper <= 0):
        raise ValueError("Lognormal approximation requires positive point and bounds")
    return (np.log(upper) - np.log(lower)) / (2 * norm.ppf(0.975))


def monte_carlo_country_trend(group: pd.DataFrame, draws: int = 4000, seed: int = SEED) -> dict:
    group = group.sort_values("year")
    x = group["year"].to_numpy(dtype=float)
    y = group["rate"].to_numpy(dtype=float)
    lower = group["lower"].to_numpy(dtype=float)
    upper = group["upper"].to_numpy(dtype=float)
    if len(np.unique(x)) < 6:
        raise ValueError("At least six distinct annual observations are required")
    sd = interval_to_log_sd(y, lower, upper)
    rng = np.random.default_rng(seed)
    simulated_log_rates = rng.normal(np.log(y), sd, size=(draws, len(y)))
    centered = x - x.mean()
    denom = np.sum(centered**2)
    slopes = simulated_log_rates @ centered / denom
    apc_draws = np.asarray(apc_from_log_slope(slopes))
    point_slope = np.sum(centered * np.log(y)) / denom
    point_apc = float(apc_from_log_slope(point_slope))
    probability = float(np.mean(apc_draws <= -2.5))
    return {
        "location": str(group["location"].iloc[0]),
        "start_year": int(x.min()),
        "end_year": int(x.max()),
        "n_years": int(len(np.unique(x))),
        "apc_point": point_apc,
        "apc_median": float(np.median(apc_draws)),
        "apc_lower": float(np.quantile(apc_draws, 0.025)),
        "apc_upper": float(np.quantile(apc_draws, 0.975)),
        "p_apc_le_minus_2_5": probability,
        "trajectory_class": classify_probability(probability),
        "uncertainty_method": "independent_lognormal_reconstruction_diagnostic",
    }


def analyse_primary(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame.query("2015 <= year <= 2023").copy()
    problems = validate_primary(primary)
    if problems:
        raise RuntimeError("Primary validation failed: " + "; ".join(problems))
    rows = []
    for index, (_, group) in enumerate(primary.groupby("location", sort=True)):
        if group["year"].nunique() < 6:
            continue
        rows.append(monte_carlo_country_trend(group, seed=SEED + index))
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No country met the six-year primary coverage rule")
    result.to_csv(RESULTS / "primary_country_trajectory_diagnostic.csv", index=False)
    return result


def analyse_who_validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subset = frame.copy()
    if "SpatialDimType" in subset:
        subset = subset[subset["SpatialDimType"].eq("COUNTRY")]
    if "Dim1" in subset:
        subset = subset[subset["Dim1"].eq("SEX_FMLE")]
    subset["NumericValue"] = pd.to_numeric(subset["NumericValue"], errors="coerce")
    subset["TimeDim"] = pd.to_numeric(subset["TimeDim"], errors="coerce")
    subset = subset.dropna(subset=["SpatialDim", "TimeDim", "NumericValue"])
    subset = subset[subset["NumericValue"] > 0]
    rows = []
    for iso3, group in subset.groupby("SpatialDim"):
        recent = group[group["TimeDim"] >= max(2010, group["TimeDim"].max() - 13)].sort_values("TimeDim")
        if recent["TimeDim"].nunique() < 6:
            continue
        x = recent["TimeDim"].to_numpy(dtype=float)
        y = recent["NumericValue"].to_numpy(dtype=float)
        centered = x - x.mean()
        slope = np.sum(centered * np.log(y)) / np.sum(centered**2)
        rows.append(
            {
                "iso3": iso3,
                "start_year": int(x.min()),
                "end_year": int(x.max()),
                "n_years": len(np.unique(x)),
                "apc_point": float(apc_from_log_slope(slope)),
                "source_status": "WHO_modelled_validation_only",
                "classification": "not_computed_without_uncertainty",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "who_validation_trends.csv", index=False)
    return result


def analyse_rate_validation(
    frame: pd.DataFrame,
    *,
    location_column: str,
    rate_column: str,
    output_name: str,
    source_label: str,
    min_year: int,
) -> pd.DataFrame:
    rows = []
    subset = frame.dropna(subset=[location_column, "year", rate_column]).copy()
    subset = subset[(subset["year"] >= min_year) & (subset[rate_column] > 0)]
    for location, group in subset.groupby(location_column):
        group = group.sort_values("year").drop_duplicates("year")
        if group["year"].nunique() < 6:
            continue
        x = group["year"].to_numpy(dtype=float)
        y = group[rate_column].to_numpy(dtype=float)
        centered = x - x.mean()
        slope = np.sum(centered * np.log(y)) / np.sum(centered**2)
        rows.append(
            {
                "location": location,
                "start_year": int(x.min()),
                "end_year": int(x.max()),
                "n_years": len(x),
                "apc_point": float(apc_from_log_slope(slope)),
                "source_status": source_label,
                "classification": "not_computed_without_source_uncertainty",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / output_name, index=False)
    return result


def data_inventory(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        year_col = next((c for c in ["year", "TimeDim"] if c in frame), None)
        location_col = next((c for c in ["location", "SpatialDim", "iso3", "country"] if c in frame), None)
        rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "locations": frame[location_col].nunique(dropna=True) if location_col else None,
                "min_year": frame[year_col].min() if year_col and not frame.empty else None,
                "max_year": frame[year_col].max() if year_col and not frame.empty else None,
                "processed_path": f"03_processed_data/{name}.csv" if name != "gbd_primary_mortality" or not frame.empty else "",
                "availability": "available" if not frame.empty else "missing_or_gated",
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(METADATA / "data_inventory.csv", index=False)
    return inventory


def process_all() -> dict[str, pd.DataFrame]:
    ensure_directories()
    who = process_who()
    who_ghe = process_who_ghe_asdr()
    who_mdb = process_who_mortality_database()
    wb_long, crosswalk = process_world_bank()
    gbd = process_gbd_primary()
    wpp = process_un_wpp()
    frames = {
        "who_mortality": who.get("who_mortality", pd.DataFrame()),
        "who_survival": who.get("who_survival", pd.DataFrame()),
        "who_ghe_breast_cancer_asdr": who_ghe,
        "who_mdb_breast_cancer_observed": who_mdb,
        "world_bank_indicators_long": wb_long,
        "country_crosswalk": crosswalk,
        "gbd_primary_mortality": gbd,
        "wpp_female_population": wpp,
    }
    data_inventory(frames)
    coverage = pd.concat([coverage_table(frame, name) for name, frame in frames.items()], ignore_index=True)
    coverage.to_csv(METADATA / "source_coverage_report.csv", index=False)
    build_source_manifest()
    return frames


def write_status(primary_available: bool, primary_complete: bool, message: str) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_data_available": primary_available,
        "primary_analysis_complete": primary_complete,
        "status": "complete" if primary_complete else "blocked_primary_data",
        "message": message,
    }
    (RESULTS / "execution_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyse_all(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    outputs = {}
    who_mortality = frames.get("who_mortality", pd.DataFrame())
    if not who_mortality.empty:
        outputs["who_validation"] = analyse_who_validation(who_mortality)
    who_ghe = frames.get("who_ghe_breast_cancer_asdr", pd.DataFrame())
    if not who_ghe.empty:
        outputs["who_ghe_validation"] = analyse_rate_validation(
            who_ghe,
            location_column="iso3",
            rate_column="asdr_per_100k",
            output_name="who_ghe_2000_2021_validation_trends.csv",
            source_label="WHO_GHE_modelled_validation_only",
            min_year=2000,
        )
    who_mdb = frames.get("who_mdb_breast_cancer_observed", pd.DataFrame())
    if not who_mdb.empty:
        outputs["who_mdb_validation"] = analyse_rate_validation(
            who_mdb,
            location_column="country",
            rate_column="asdr_per_100k",
            output_name="who_mdb_2015_latest_observed_validation_trends.csv",
            source_label="WHO_MDB_observed_validation_only",
            min_year=2015,
        )
    gbd = frames.get("gbd_primary_mortality", pd.DataFrame())
    if gbd.empty:
        write_status(False, False, "GBD 2023 primary export is absent; see 01_metadata/MANUAL_DOWNLOAD_REQUIRED.md")
        return outputs
    outputs["primary"] = analyse_primary(gbd)
    hierarchical = analyse_hierarchical_trajectory(gbd)
    if not hierarchical.empty:
        outputs["hierarchical"] = hierarchical
    # WHO regions as an alternative pooling structure. Reported as a sensitivity analysis, not as a
    # replacement: the frozen protocol names region without naming a regional taxonomy, and the
    # World Bank grouping also carries the income classification used elsewhere.
    who_region_pooled = analyse_hierarchical_trajectory(
        gbd, grouping="who_region", output_name="hierarchical_country_trajectory_who_region.csv"
    )
    if not who_region_pooled.empty:
        outputs["hierarchical_who_region"] = who_region_pooled
    subgroups = analyse_subgroups(gbd, hierarchical)
    if subgroups:
        outputs["subgroup_summary"] = subgroups["summary"]
        outputs["subgroup_assignments"] = subgroups["assignments"]
        outputs["subgroup_heterogeneity"] = subgroups["heterogeneity"]
        aggregate_check = validate_who_region_aggregates(subgroups["assignments"])
        if not aggregate_check.empty:
            outputs["who_region_aggregate_validation"] = aggregate_check
    lags = analyse_health_system_lags(gbd)
    if lags:
        outputs["health_system_lags"] = lags["coefficients"]
        outputs["health_system_summary"] = lags["summary"]
    backtest = backtest_forecast(gbd)
    if not backtest.empty:
        outputs["forecast_backtest"] = backtest
    population = frames.get("wpp_female_population", pd.DataFrame())
    forecast = forecast_scenarios(gbd, population)
    if not forecast.empty:
        outputs["forecast"] = forecast
    decomposition = decompose_death_change()
    if not decomposition.empty:
        outputs["decomposition"] = decomposition
    write_status(
        True,
        True,
        "Trajectory classification (diagnostic and partially pooled confirmatory), prespecified "
        "subgroup stratification including WHO regions, health-system distributed-lag association, "
        "back-tested 2040 scenarios and the four-factor decomposition are complete; the "
        "predictor-circularity audit of the WHO survival estimates has run and admits their source "
        "class as the seventh stratification axis while rejecting the survival value as an "
        "explanatory variable; manuscript draft v1 is written and passes the numeric audit in "
        "04_code/audit_manuscript.py",
    )
    return outputs


def render_outputs(frames: dict[str, pd.DataFrame], analyses: dict[str, pd.DataFrame]) -> None:
    ensure_directories()
    coverage_path = METADATA / "source_coverage_report.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        coverage.to_csv(TABLES / "table_s1_source_coverage.csv", index=False)
    survival = frames.get("who_survival", pd.DataFrame())
    if not survival.empty:
        country = survival[survival.get("SpatialDimType", pd.Series(index=survival.index, dtype=str)).eq("COUNTRY")].copy()
        columns = [c for c in ["SpatialDim", "TimeDimensionValue", "NumericValue", "Low", "High", "source_status"] if c in country]
        country[columns].to_csv(TABLES / "table_s2_who_breast_cancer_survival.csv", index=False)
    if "primary" in analyses and not analyses["primary"].empty:
        result = analyses["primary"].sort_values("p_apc_le_minus_2_5", ascending=False)
        result.to_csv(TABLES / "table_1_country_trajectory_probabilities.csv", index=False)
        _plot_probability(result)
    if "hierarchical" in analyses and not analyses["hierarchical"].empty:
        hierarchical = analyses["hierarchical"]
        hierarchical.to_csv(TABLES / "table_3_hierarchical_trajectory.csv", index=False)
        _plot_shrinkage(hierarchical, analyses.get("primary", pd.DataFrame()))
    if "hierarchical_who_region" in analyses and not analyses["hierarchical_who_region"].empty:
        analyses["hierarchical_who_region"].to_csv(
            TABLES / "table_s4_hierarchical_trajectory_who_region.csv", index=False
        )
    if "subgroup_summary" in analyses and not analyses["subgroup_summary"].empty:
        analyses["subgroup_summary"].to_csv(TABLES / "table_6_subgroup_stratification.csv", index=False)
        _plot_subgroups(analyses["subgroup_summary"])
        if "subgroup_assignments" in analyses and not analyses["subgroup_assignments"].empty:
            _plot_trajectory_map(analyses["subgroup_assignments"])
    if "who_region_aggregate_validation" in analyses and not analyses["who_region_aggregate_validation"].empty:
        analyses["who_region_aggregate_validation"].to_csv(
            TABLES / "table_s5_who_region_aggregate_validation.csv", index=False
        )
    if "health_system_lags" in analyses and not analyses["health_system_lags"].empty:
        lags = analyses["health_system_lags"]
        lags.to_csv(TABLES / "table_2_health_system_lag_coefficients.csv", index=False)
        _plot_distributed_lag(lags, analyses.get("health_system_summary", pd.DataFrame()))
    if "forecast" in analyses and not analyses["forecast"].empty:
        forecast = analyses["forecast"]
        world = forecast.groupby(["scenario", "year"], as_index=False)["deaths"].sum()
        world.to_csv(TABLES / "table_4_forecast_scenarios.csv", index=False)
        _plot_forecast(world, analyses.get("forecast_backtest", pd.DataFrame()))
    if "decomposition" in analyses and not analyses["decomposition"].empty:
        analyses["decomposition"].to_csv(TABLES / "table_5_decomposition.csv", index=False)
        _plot_decomposition(analyses["decomposition"])
    validation_frames = []
    if "who_ghe_validation" in analyses and not analyses["who_ghe_validation"].empty:
        frame = analyses["who_ghe_validation"].copy()
        frame["validation_source"] = "WHO GHE modelled, 2000-2021 benchmark years"
        validation_frames.append(frame)
    if "who_mdb_validation" in analyses and not analyses["who_mdb_validation"].empty:
        frame = analyses["who_mdb_validation"].copy()
        frame["validation_source"] = "WHO MDB observed, 2015-latest"
        validation_frames.append(frame)
    if validation_frames:
        validation = pd.concat(validation_frames, ignore_index=True)
        validation.to_csv(TABLES / "table_s3_validation_trends_no_classification.csv", index=False)
        _plot_validation_apc(validation)
    status = json.loads((RESULTS / "execution_status.json").read_text(encoding="utf-8"))
    report = [
        "# Execution report",
        "",
        f"Generated: {status['generated_utc']}",
        "",
        f"Primary status: **{status['status']}**",
        "",
        status["message"],
        "",
        "WHO validation results are not substitutes for the primary GBD analysis and do not include trajectory classifications when source uncertainty is unavailable.",
    ]
    (RESULTS / "execution_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not status["primary_analysis_complete"]:
        blocker = [
            "# Primary-analysis blocker report",
            "",
            "## Blocking input",
            "",
            "The authenticated IHME GBD 2023 breast-cancer export is not present. It is the protocol-defined primary outcome source.",
            "",
            "## Why execution stopped",
            "",
            "Replacing it with WHO GHE or observed-country data would change coverage, uncertainty, and the primary estimand. The protocol therefore prohibits country target classification from substitute data.",
            "",
            "## Work completed despite the blocker",
            "",
            "Public WHO mortality, WHO survival, WHO GHE, WHO Mortality Database, and World Bank inputs were retrieved and checksummed. Validation-only trends were produced without attainment labels.",
            "",
            "## Resolution",
            "",
            "Follow `01_metadata/MANUAL_DOWNLOAD_REQUIRED.md`, place the untouched export in `02_raw_data/ihme_gbd/gbd_2023/`, and rerun `python 04_code/run_pipeline.py all`.",
        ]
        (RESULTS / "BLOCKER_REPORT.md").write_text("\n".join(blocker) + "\n", encoding="utf-8")


def _plot_probability(result: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    show = pd.concat([result.head(15), result.tail(15)]).drop_duplicates("location")
    show = show.sort_values("p_apc_le_minus_2_5")
    colours = show["trajectory_class"].map({"on_trajectory": "#17823b", "uncertain": "#d89000", "off_trajectory": "#b72c2c"})
    fig, ax = plt.subplots(figsize=(9, max(6, 0.27 * len(show))))
    ax.barh(show["location"], show["p_apc_le_minus_2_5"], color=colours)
    ax.axvline(0.2, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability APC ≤ -2.5%")
    ax.set_title("Breast-cancer mortality trajectory probability (selected extremes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_trajectory_probability.png", dpi=200)
    fig.savefig(FIGURES / "figure_1_trajectory_probability.svg")
    plt.close(fig)


def _plot_forecast(world: pd.DataFrame, backtest: pd.DataFrame) -> None:
    """Global deaths to 2040 under the three scenarios."""
    import matplotlib.pyplot as plt

    labels = {
        "continuation": ("Continuation of current country trends", "#b72c2c"),
        "stagnation": ("Rates held at 2023", "#d89000"),
        "benchmark_2_5pc": ("Every country meets -2.5% per year", "#17823b"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for scenario, (label, colour) in labels.items():
        subset = world[world["scenario"].eq(scenario)].sort_values("year")
        if subset.empty:
            continue
        ax.plot(subset["year"], subset["deaths"] / 1e6, label=label, color=colour, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Female breast-cancer deaths (millions)")
    title = "Projected global breast-cancer deaths to 2040 under three scenarios"
    if not backtest.empty:
        median_error = float(backtest["mean_absolute_percentage_error"].median())
        title += f"\nback-test on held-out 2020-2023: median absolute error {median_error:.1f}%"
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.text(
        0.01,
        0.01,
        "Age-specific rates applied to WPP medium-variant female population. Scenario contrast "
        "under a stated proportional assumption; not a prediction and not avoidable deaths.",
        fontsize=7,
        color="#64748b",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "figure_4_forecast_scenarios.png", dpi=200)
    fig.savefig(FIGURES / "figure_4_forecast_scenarios.svg")
    plt.close(fig)


def _plot_decomposition(summary: pd.DataFrame) -> None:
    """Waterfall of the four factors driving the change in deaths."""
    import matplotlib.pyplot as plt

    row = summary.iloc[0]
    components = [
        ("Population size", float(row["population_effect"])),
        ("Age structure", float(row["age_structure_effect"])),
        ("Incidence rate", float(row["incidence_effect"])),
        ("Mortality-to-\nincidence ratio", float(row["mortality_incidence_ratio_effect"])),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Track the running total so the axis can be sized to fit the value labels; without headroom
    # the tallest label collides with the title.
    running = 0.0
    tops = [0.0]
    for _, value in components:
        running += value
        tops.append(running)
    span = max(tops) - min(tops)
    pad = max(span * 0.08, 1.0) / 1000.0

    running = 0.0
    for index, (label, value) in enumerate(components):
        colour = "#b72c2c" if value > 0 else "#17823b"
        ax.bar(index, value / 1000.0, bottom=running / 1000.0, color=colour, width=0.62)
        top = (running + value) / 1000.0
        ax.text(
            index,
            top + pad * 0.25 if value > 0 else top - pad * 0.55,
            f"{value / 1000.0:+.1f}k",
            ha="center",
            fontsize=9,
        )
        running += value
    ax.bar(len(components), running / 1000.0, color="#1c4966", width=0.62)
    ax.text(len(components), running / 1000.0 + pad * 0.25, f"{running / 1000.0:+.1f}k", ha="center", fontsize=9)
    ax.set_ylim(min(min(tops), 0.0) / 1000.0 - pad, max(tops) / 1000.0 + pad * 1.6)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(range(len(components) + 1))
    ax.set_xticklabels([label for label, _ in components] + ["Net change"], fontsize=9)
    ax.set_ylabel("Contribution to change in annual deaths (thousands)")
    ax.set_title(
        f"What drove the change in breast-cancer deaths, {int(row['start_year'])}-{int(row['end_year'])}\n"
        f"{int(row['deaths_start']):,} to {int(row['deaths_end']):,} deaths across {int(row['countries'])} countries",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "Exact LMDI decomposition. The fourth factor is a crude mortality-to-incidence ratio, not survival.",
        fontsize=7,
        color="#64748b",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "figure_5_decomposition.png", dpi=200)
    fig.savefig(FIGURES / "figure_5_decomposition.svg")
    plt.close(fig)


def _plot_shrinkage(hierarchical: pd.DataFrame, diagnostic: pd.DataFrame) -> None:
    """Unpooled against partially pooled annual percent change, by pooling group."""
    import matplotlib.pyplot as plt

    frame = hierarchical.dropna(subset=["apc_unpooled", "apc_pooled"])
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    groups = sorted(frame["group"].unique())
    palette = plt.get_cmap("tab10")
    for index, group in enumerate(groups):
        subset = frame[frame["group"].eq(group)]
        ax.scatter(
            subset["apc_unpooled"],
            subset["apc_pooled"],
            s=26,
            alpha=0.85,
            color=palette(index % 10),
            label=f"{group} (n={len(subset)})",
            edgecolor="white",
            linewidth=0.4,
        )
    limits = [
        float(min(frame["apc_unpooled"].min(), frame["apc_pooled"].min())) - 0.5,
        float(max(frame["apc_unpooled"].max(), frame["apc_pooled"].max())) + 0.5,
    ]
    ax.plot(limits, limits, color="#94a3b8", linewidth=1, linestyle="--", label="no shrinkage")
    ax.axhline(-2.5, color="#b72c2c", linewidth=1)
    ax.axvline(-2.5, color="#b72c2c", linewidth=1, alpha=0.4)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Unpooled annual percent change, 2015-2023 (%)")
    ax.set_ylabel("Partially pooled annual percent change (%)")
    ax.set_title(
        "Partial pooling of country mortality trends\n"
        "points fall inside the dashed line because noisy country slopes shrink to their group",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.text(
        0.01,
        0.01,
        "Red lines mark the -2.5% per year GBCI benchmark. Empirical-Bayes partial pooling; "
        "reconstructed intervals cannot recover cross-year correlation.",
        fontsize=7,
        color="#64748b",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "figure_3_partial_pooling.png", dpi=200)
    fig.savefig(FIGURES / "figure_3_partial_pooling.svg")
    plt.close(fig)


def _plot_distributed_lag(coefficients: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Distributed-lag coefficients for out-of-pocket share, with the cumulative effect stated."""
    import matplotlib.pyplot as plt

    lag_rows = coefficients[coefficients["term"].str.startswith("oop_lag")].copy()
    if lag_rows.empty:
        return
    lag_rows["lag"] = lag_rows["term"].str.replace("oop_lag", "", regex=False).astype(int)
    lag_rows = lag_rows.sort_values("lag")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        lag_rows["lag"],
        lag_rows["coefficient"],
        yerr=[
            lag_rows["coefficient"] - lag_rows["ci_low"],
            lag_rows["ci_high"] - lag_rows["coefficient"],
        ],
        fmt="o",
        capsize=4,
        color="#1c4966",
        ecolor="#7f9fb2",
    )
    ax.axhline(0.0, color="#b72c2c", linewidth=1)
    ax.set_xticks(list(lag_rows["lag"]))
    ax.set_xlabel("Lag (years) of out-of-pocket share of current health expenditure")
    ax.set_ylabel("Change in log age-standardised mortality per 1 percentage point")
    title = "Distributed-lag association of out-of-pocket share with breast-cancer mortality"
    if not summary.empty:
        row = summary.iloc[0]
        title += (
            f"\ncumulative lags 3-7: {row['coefficient']:+.4f} (p = {row['p_value']:.3f}); "
            f"{int(row['countries'])} countries, {int(row['observations'])} observations"
        )
    ax.set_title(title, fontsize=10)
    fig.text(
        0.01,
        0.01,
        "Two-way fixed effects with country-clustered standard errors. Association only; not a causal effect.",
        fontsize=7,
        color="#64748b",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "figure_2_health_system_distributed_lag.png", dpi=200)
    fig.savefig(FIGURES / "figure_2_health_system_distributed_lag.svg")
    plt.close(fig)


def _plot_validation_apc(validation: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    sources = list(validation["validation_source"].drop_duplicates())
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5), squeeze=False)
    for axis, source in zip(axes[0], sources):
        values = validation.loc[validation["validation_source"].eq(source), "apc_point"].dropna()
        axis.hist(values, bins=24, color="#34699a", edgecolor="white")
        axis.axvline(-2.5, color="#b72c2c", linestyle="--", label="-2.5% benchmark")
        axis.set_title(source)
        axis.set_xlabel("Point-estimate APC (%)")
        axis.set_ylabel("Countries")
        axis.legend()
    fig.suptitle("Validation-source breast-cancer mortality trends (no attainment classification)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_s1_validation_apc_distributions.png", dpi=200)
    fig.savefig(FIGURES / "figure_s1_validation_apc_distributions.svg")
    plt.close(fig)


def _forest_column(subfigure, summary: pd.DataFrame, axes_present: list[str], overall_apc: float, limits: tuple[float, float]) -> None:
    """Stacked forest panels sharing one horizontal scale, drawn into a sub-figure."""
    blocks = [summary[summary["axis"].eq(axis)].sort_values("apc_mean") for axis in axes_present]
    panels = subfigure.subplots(
        len(blocks), 1, sharex=True, gridspec_kw={"height_ratios": [max(1, len(b)) for b in blocks]}, squeeze=False
    )[:, 0]
    for panel, block in zip(panels, blocks):
        positions = np.arange(len(block))
        # Colour marks the direction of the stratum trend. Marking distance from the benchmark
        # would be a constant here, because no stratum mean approaches a 2.5% annual decline.
        colours = [
            "#17823b" if upper < 0 else "#b72c2c" if lower > 0 else "#d89000"
            for lower, upper in zip(block["apc_lower"], block["apc_upper"])
        ]
        panel.hlines(positions, block["apc_lower"], block["apc_upper"], color=colours, linewidth=2.4)
        panel.scatter(block["apc_mean"], positions, color=colours, s=30, zorder=3)
        panel.axvline(-2.5, color="#334155", linestyle="--", linewidth=1)
        panel.axvline(0, color="#cbd5e1", linewidth=0.8, zorder=0)
        if np.isfinite(overall_apc):
            panel.axvline(overall_apc, color="#94a3b8", linestyle=":", linewidth=1)
        panel.set_yticks(positions)
        panel.set_yticklabels(
            [f"{level}  (n={int(count)})" for level, count in zip(block["level"], block["countries"])],
            fontsize=8,
        )
        panel.set_ylim(-0.65, len(block) - 0.35)
        panel.set_xlim(*limits)
        panel.set_title(str(block["axis_label"].iloc[0]), fontsize=9.5, loc="left")
        panel.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            panel.spines[spine].set_visible(False)
    panels[-1].set_xlabel("Mean annual percentage change (%)", fontsize=9)


def _plot_subgroups(summary: pd.DataFrame) -> None:
    """Forest plot of stratum mean annual percentage change, one panel per stratification axis.

    All panels share one horizontal scale so strata on different axes can be compared directly,
    which independent per-panel scaling would prevent.
    """
    import matplotlib.pyplot as plt

    present = {axis for axis in summary["axis"]}
    left_axes = [axis for axis in ("who_region", "world_bank_region") if axis in present]
    right_axes = [
        axis
        for axis in (
            "income_group",
            "baseline_mortality_tertile",
            "outcome_precision_tertile",
            "registry_series",
            "survival_source_class",
        )
        if axis in present
    ]
    if not left_axes and not right_axes:
        return
    overall = summary[summary["axis"].eq("overall")]
    overall_apc = float(overall["apc_mean"].iloc[0]) if not overall.empty else float("nan")
    strata = summary[~summary["axis"].eq("overall")]
    low = min(float(strata["apc_lower"].min()), -2.5)
    high = float(strata["apc_upper"].max())
    pad = 0.08 * (high - low)
    limits = (low - pad, high + pad)

    figure = plt.figure(figsize=(13.2, 8.4))
    columns = figure.subfigures(1, 2, wspace=0.04)
    for subfigure, axes_present in zip(columns, (left_axes, right_axes)):
        if axes_present:
            _forest_column(subfigure, summary, axes_present, overall_apc, limits)
    figure.suptitle(
        "Breast-cancer mortality trend by prespecified stratum, 2015-2023\n"
        "Dashed line is the -2.5% benchmark; dotted line is the all-country mean; grey line is no change",
        fontsize=11,
    )
    figure.text(
        0.005,
        0.008,
        "Random-effects means of unpooled country slopes, each weighted by its own precision. Green marks a stratum whose "
        "95% interval lies wholly below zero, red wholly above zero, amber spanning zero. No stratum mean reaches the benchmark.",
        fontsize=7,
        color="#64748b",
    )
    figure.savefig(FIGURES / "figure_6_subgroup_stratification.png", dpi=200, bbox_inches="tight")
    figure.savefig(FIGURES / "figure_6_subgroup_stratification.svg", bbox_inches="tight")
    plt.close(figure)


def _world_boundaries() -> list[dict]:
    """Natural Earth 1:110m country polygons, or an empty list when the snapshot is absent."""
    base = latest_snapshot_dir("natural_earth")
    if base is None:
        return []
    path = base / "ne_110m_admin_0_countries.geojson"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("features", [])


def _plot_trajectory_map(assignments: pd.DataFrame) -> None:
    """World map of confirmatory trajectory class.

    Drawn in plate carree, which distorts area at high latitude. The map is a locator for which
    countries fall in which class and is not a quantitative display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPolygon

    features = _world_boundaries()
    if not features:
        return
    palette = {
        "on_trajectory": "#17823b",
        "uncertain": "#d89000",
        "off_trajectory": "#b72c2c",
    }
    labels = {
        "on_trajectory": "On trajectory (probability at least 0.80)",
        "uncertain": "Uncertain (0.20 to 0.799)",
        "off_trajectory": "Off trajectory (below 0.20)",
    }
    classes = (
        assignments.dropna(subset=["iso3", "trajectory_class"])
        .drop_duplicates("iso3")
        .set_index("iso3")["trajectory_class"]
        .to_dict()
    )
    figure, panel = plt.subplots(figsize=(13, 6.6))
    drawn, unmatched = set(), 0
    for feature in features:
        properties = feature.get("properties", {})
        iso3 = properties.get("ISO_A3_EH") or properties.get("ADM0_ISO") or properties.get("ISO_A3")
        trajectory = classes.get(iso3)
        if trajectory is None:
            unmatched += 1
        else:
            drawn.add(iso3)
        colour = palette.get(trajectory, "#e2e8f0")
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            rings = [geometry["coordinates"][0]]
        elif geometry.get("type") == "MultiPolygon":
            rings = [part[0] for part in geometry["coordinates"]]
        else:
            continue
        for ring in rings:
            panel.add_patch(
                MplPolygon(
                    np.asarray(ring, dtype=float),
                    closed=True,
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=0.28,
                )
            )
    panel.set_xlim(-180, 180)
    panel.set_ylim(-60, 85)
    panel.set_aspect("equal")
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=10, color=palette[key], label=labels[key])
        for key in ("on_trajectory", "uncertain", "off_trajectory")
    ]
    handles.append(
        Line2D([], [], marker="s", linestyle="none", markersize=10, color="#e2e8f0", label="Not analysed or not drawn at this scale")
    )
    panel.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, ncol=2)
    panel.set_title(
        "Probability of meeting the 2.5% annual reduction benchmark, 2015-2023\n"
        f"Confirmatory partially pooled classification; {len(drawn)} of {assignments['iso3'].nunique()} "
        "analysed countries are drawn at 1:110m",
        fontsize=11,
    )
    figure.text(
        0.01,
        0.01,
        "Plate carree projection, which distorts area away from the equator. Small states are not "
        "visible at this scale and are reported in the country table.",
        fontsize=7,
        color="#64748b",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(FIGURES / "figure_7_trajectory_map.png", dpi=200)
    figure.savefig(FIGURES / "figure_7_trajectory_map.svg")
    plt.close(figure)
