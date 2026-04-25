"""Load and cache enrichment CSVs (academic calendar, sports, disruptions) for date-based lookup."""

import pandas as pd
from .config import DATA_DIR

_calendar = None
_sports = None
_disruptions = None
_events = None
_semester_metadata = None


def _load_calendar():
    global _calendar, _semester_metadata
    if _calendar is not None:
        return _calendar

    cal = pd.read_csv(DATA_DIR / "academic_calendar.csv")
    cal["date"] = cal["date"].astype(str)
    finals_mask = cal["category"] == "finals"
    _semester_metadata = {
        "first_class_date": cal[cal["category"] == "semester_start"]["date"].min(),
        "finals_start_date": cal[finals_mask]["date"].min(),
        "total_weeks": 16,
    }
    lookup = {}
    for _, row in cal.iterrows():
        cat = row.get("category", "")
        lookup[row["date"]] = {
            "is_class_day": int(row["is_class_day"]) if not pd.isna(row["is_class_day"]) else 0,
            "is_break": int(cat == "spring_recess"),
            "is_finals": int(cat == "finals"),
            "is_commencement": int(cat == "commencement"),
            "is_holiday": int(cat == "university_closed"),
        }
    _calendar = lookup
    return _calendar


def _load_sports():
    global _sports
    if _sports is not None:
        return _sports

    sports = pd.read_csv(DATA_DIR / "sports_schedule.csv")
    sports["date"] = sports["date"].astype(str)
    home = sports[sports["home_away"] == "home"]

    lookup = {}
    for date, group in home.groupby("date"):
        sports_list = group["sport"].tolist()
        impacts = group["parking_impact"].tolist()
        lookup[date] = {
            "home_game_count": len(sports_list),
            "has_basketball": int(any("Basketball" in s for s in sports_list)),
            "has_baseball": int(any("Baseball" in s for s in sports_list)),
            "has_softball": int(any("Softball" in s for s in sports_list)),
            "has_lacrosse": int(any("Lacrosse" in s for s in sports_list)),
            "high_impact_game": int(any(v == "high" for v in impacts)),
        }
    _sports = lookup
    return _sports


def _load_disruptions():
    global _disruptions
    if _disruptions is not None:
        return _disruptions

    dis = pd.read_csv(DATA_DIR / "campus_disruptions.csv")
    dis["date"] = dis["date"].astype(str)
    condition_map = {"Normal": 0, "C1": 1, "C2": 2, "C2->Normal": 1, "C1->C2": 2}

    lookup = {}
    for _, row in dis.iterrows():
        classes = str(row.get("classes", ""))
        lookup[row["date"]] = {
            "condition_level": condition_map.get(row.get("condition", "Normal"), 0),
            "is_remote": int("remote" in classes),
            "is_cancelled": int("cancelled" in classes),
        }
    _disruptions = lookup
    return _disruptions


def _load_events():
    global _events
    if _events is not None:
        return _events

    impact_level = {"none": 0, "low": 1, "medium": 2, "high": 3}
    events = pd.read_csv(DATA_DIR / "campus_events.csv")
    events["date"] = events["date"].astype(str)
    events["impact_num"] = events["parking_impact"].map(impact_level).fillna(0).astype(int)

    lookup = {}
    for date, group in events.groupby("date"):
        impacts = group["impact_num"].tolist()
        lookup[date] = {
            "event_max_impact": max(impacts) if impacts else 0,
            "event_high_count": int(sum(impact == 3 for impact in impacts)),
        }
    _events = lookup
    return _events


def get_calendar(date_str: str) -> dict:
    cal = _load_calendar()
    return cal.get(date_str, {
        "is_class_day": 1,
        "is_break": 0,
        "is_finals": 0,
        "is_commencement": 0,
        "is_holiday": 0,
    })


def get_sports(date_str: str) -> dict:
    sports = _load_sports()
    return sports.get(date_str, {
        "home_game_count": 0,
        "has_basketball": 0,
        "has_baseball": 0,
        "has_softball": 0,
        "has_lacrosse": 0,
        "high_impact_game": 0,
    })


def get_disruptions(date_str: str) -> dict:
    dis = _load_disruptions()
    return dis.get(date_str, {
        "condition_level": 0,
        "is_remote": 0,
        "is_cancelled": 0,
    })


def get_events(date_str: str) -> dict:
    events = _load_events()
    return events.get(date_str, {
        "event_max_impact": 0,
        "event_high_count": 0,
    })


def get_semester_metadata() -> dict:
    _load_calendar()
    return dict(_semester_metadata or {})


def get_coverage() -> dict:
    cal = _load_calendar()
    sports = _load_sports()
    dis = _load_disruptions()
    events = _load_events()
    return {
        "academic_calendar": {"min": min(cal.keys()), "max": max(cal.keys()), "entries": len(cal)} if cal else None,
        "sports_schedule": {"min": min(sports.keys()), "max": max(sports.keys()), "entries": len(sports)} if sports else None,
        "campus_disruptions": {"min": min(dis.keys()), "max": max(dis.keys()), "entries": len(dis)} if dis else None,
        "campus_events": {"min": min(events.keys()), "max": max(events.keys()), "entries": len(events)} if events else None,
    }
