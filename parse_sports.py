#!/usr/bin/env python3
"""
Parse raw Charlotte Athletics composite calendar text into sports_schedule CSV format.

The raw text format uses date headers followed by event lines:

    Saturday, April 4, 2026
    Women's Lacrosse vs James Madison 5:00 PM     <- home (vs + Charlotte location)
    Charlotte, NC                                  <- location sometimes on next line
    Jerry Richardson Stadium                       <- sub-venue, ignored
    Baseball at East Carolina 1:00 PM              <- away (at)
    Greenville, N.C.
    Men's Tennis at #70 South Florida 11:00 AM Tampa, Fl.  <- inline location

Home/away is determined by "vs" vs "at" + whether the location is Charlotte.
Parking impact is assigned by sport (home only; away = none).

Usage:
    python parse_sports.py [raw_file] [csv_file]

Defaults:
    raw_file = data/raw_sports_april_may.txt
    csv_file = data/sports_schedule.csv

New rows are appended (duplicates by date+sport+opponent are skipped).
"""

import re
import csv
import sys
from datetime import datetime
from pathlib import Path

# --- Sport definitions -------------------------------------------------

SPORTS = sorted([
    "Men's Basketball", "Women's Basketball",
    "Baseball", "Softball",
    "Women's Lacrosse",
    "Men's Soccer", "Women's Soccer",
    "Men's Tennis", "Women's Tennis",
    "Track & Field",
    "Men's Golf", "Women's Golf",
], key=len, reverse=True)  # longest first to avoid partial matches

# Parking impact for HOME games only; away games are always "none"
HOME_IMPACT = {
    "Men's Basketball":  "high",
    "Women's Basketball": "high",
    "Baseball":          "medium",
    "Softball":          "medium",
    "Women's Lacrosse":  "medium",
    "Men's Soccer":      "low",
    "Women's Soccer":    "low",
    "Men's Tennis":      "low",
    "Women's Tennis":    "low",
    "Track & Field":     "none",
    "Men's Golf":        "none",
    "Women's Golf":      "none",
}

# --- Regex patterns ----------------------------------------------------

# Date header: "Saturday, April 4, 2026"
DATE_RE = re.compile(
    r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
    r',\s+(\w+)\s+(\d{1,2}),\s+(\d{4})$',
    re.IGNORECASE,
)

# Sport event line: "{Sport} vs/at {rest}"
# Longest-first sport list prevents "Men's Tennis" matching before "Men's Tennis" etc.
SPORT_RE = re.compile(
    r'^(' + '|'.join(re.escape(s) for s in SPORTS) + r')\s+(vs\.?|at)\s+(.*)',
    re.IGNORECASE,
)

# Time within the "rest" portion of an event line
TIME_RE = re.compile(r'\b(\d{1,2}:\d{2}\s*[AP]M|TBD|TBA)\b', re.IGNORECASE)

# Lines that are clearly noise and should be skipped
NOISE_RE = re.compile(
    r'^('
    r'[WL]\s+\d+[-–]\d+'           # scores: W 4-1, L 2-12
    r'|\(\d+'                       # extra info: (7), (5 Inn.)
    r'|\d+(st|nd|rd|th)\s+of\s+\d+'# standings: 7th of 15
    r'|[-–]\s+'                     # sub-notes: - Outdoor Season
    r'|Jerry Richardson Stadium'
    r'|Hayes Stadium'
    r'|Charlotte Soccer Field'
    r'|Charlotte Athletics'
    r'|Composite Calendar'
    r')',
    re.IGNORECASE,
)

MONTH_MAP = {
    'january': 1, 'february': 2,  'march': 3,    'april': 4,
    'may': 5,     'june': 6,      'july': 7,      'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


# --- Parsing helpers ---------------------------------------------------

def is_charlotte(location: str) -> bool:
    return 'charlotte' in location.lower()


def parse_event_line(line: str):
    """
    Match a sport event line. Returns (sport, vs_at, opponent, time, location_inline)
    or None if the line isn't a sport event.

    location_inline is empty string when the location is on the following line.
    """
    m = SPORT_RE.match(line)
    if not m:
        return None

    sport, vs_at, rest = m.group(1), m.group(2).lower().rstrip('.'), m.group(3).strip()

    time_m = TIME_RE.search(rest)
    if time_m:
        time = time_m.group(1).upper()
        opponent = rest[:time_m.start()].strip().rstrip(',').strip()
        location_inline = rest[time_m.end():].strip()
    else:
        time = ''
        opponent = rest.strip()
        location_inline = ''

    # Strip ranking prefix from opponent: "#70 South Florida" -> "South Florida"
    opponent = re.sub(r'^#\d+\s+', '', opponent)

    return sport, vs_at, opponent, time, location_inline


def resolve_home_away(vs_at: str, location: str) -> str:
    if vs_at == 'at':
        return 'away'
    # "vs" — home only if the location is Charlotte
    if location and is_charlotte(location):
        return 'home'
    if location:
        return 'away'
    # "vs" with no location resolved — assume home (scraper default)
    return 'home'


# --- Main parser -------------------------------------------------------

def parse_file(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    events = []

    current_date = None
    current_dow = None
    pending = None  # event awaiting location resolution

    def flush(location_override=None):
        nonlocal pending
        if pending is None:
            return
        loc = location_override or pending['location']
        home_away = resolve_home_away(pending['vs_at'], loc)
        impact = HOME_IMPACT.get(pending['sport'], 'none') if home_away == 'home' else 'none'
        events.append({
            'date':           pending['date'],
            'day_of_week':    pending['dow'],
            'sport':          pending['sport'],
            'home_away':      home_away,
            'opponent':       pending['opponent'],
            'time':           pending['time'],
            'location':       loc,
            'parking_impact': impact,
        })
        pending = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip noise lines
        if NOISE_RE.match(line):
            continue

        # Date header
        dm = DATE_RE.match(line)
        if dm:
            flush()
            day_name, month_str, day_str, year_str = dm.groups()
            dt = datetime(int(year_str), MONTH_MAP[month_str.lower()], int(day_str))
            current_date = dt.strftime('%Y-%m-%d')
            current_dow = day_name
            continue

        # Sport event line
        parsed = parse_event_line(line)
        if parsed:
            flush()
            sport, vs_at, opponent, time, location_inline = parsed
            pending = {
                'date':     current_date,
                'dow':      current_dow,
                'sport':    sport,
                'vs_at':    vs_at,
                'opponent': opponent,
                'time':     time,
                'location': location_inline,
            }
            continue

        # Location continuation: attach to pending event if it has no location yet
        if pending is not None and not pending['location']:
            pending['location'] = line
            continue

        # Unrecognized line (golf rounds without vs/at, multi-line venue names, etc.) — ignore

    flush()  # flush any trailing event

    # Deduplicate: the raw file has two overlapping sections
    seen = set()
    unique = []
    for e in events:
        key = (e['date'], e['sport'], e['opponent'])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# --- CSV output --------------------------------------------------------

FIELDNAMES = ['date', 'day_of_week', 'sport', 'home_away', 'opponent',
              'time', 'location', 'parking_impact']


def load_existing_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline='') as f:
        return {(r['date'], r['sport'], r['opponent']) for r in csv.DictReader(f)}


def append_to_csv(csv_path: Path, rows: list[dict]):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for row in rows:
            writer.writerow(row)


# --- Entry point -------------------------------------------------------

def main():
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/raw_sports_april_may.txt')
    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/sports_schedule.csv')

    print(f"Parsing {raw_path} ...")
    all_events = parse_file(raw_path)
    print(f"  {len(all_events)} unique events found")

    existing_keys = load_existing_keys(csv_path)
    new_rows = [e for e in all_events if (e['date'], e['sport'], e['opponent']) not in existing_keys]
    skipped = len(all_events) - len(new_rows)

    print(f"  {skipped} already in CSV, {len(new_rows)} new")

    if not new_rows:
        print("Nothing to add.")
        return

    append_to_csv(csv_path, new_rows)
    print(f"  Appended {len(new_rows)} rows to {csv_path}")

    # Summary of home games added
    home = [e for e in new_rows if e['home_away'] == 'home']
    print(f"\nHome games added ({len(home)}):")
    for e in home:
        print(f"  {e['date']}  {e['sport']:<22}  vs {e['opponent']:<30}  {e['time']:<10}  impact={e['parking_impact']}")

    away = [e for e in new_rows if e['home_away'] == 'away']
    print(f"\nAway/neutral games added: {len(away)}")


if __name__ == '__main__':
    main()
