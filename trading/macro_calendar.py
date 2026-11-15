"""Public scheduled-event calendar with explicit provenance and partial coverage."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import html
import json
import re
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'bls':'https://www.bls.gov/schedule/news_release/bls.ics',
    'bea':'https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics',
    'fed':'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
    'ff_week':'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
    'ff_next':'https://nfs.faireconomy.media/ff_calendar_nextweek.json',
}


def utc(value=None):
    stamp=pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:raise ValueError('Timezone required')
    return stamp.tz_convert('UTC')


def event_kind(title):
    t=title.lower()
    if 'personal income and outlays' in t or 'pce price' in t:return 'PCE'
    if 'consumer price index' in t or re.search(r'\bcpi\b',t):return 'CPI'
    if 'employment situation' in t or 'non-farm employment change' in t or 'nonfarm payroll' in t:return 'NFP'
    if ('fomc' in t and any(x in t for x in ['statement','rate','press conference'])) or 'federal funds rate' in t:return 'FOMC'
    return None


def cached_get(url,now,cache_dir=None):
    folder=Path(cache_dir) if cache_dir else ROOT/'logs/context_calendar'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    if path.exists():
        try:
            old=json.loads(path.read_text(encoding='utf-8'))
            age=(now-utc(old['fetched_at'])).total_seconds()
            if 0<=age<3600:return old['text'],old['fetched_at']
        except (ValueError,KeyError):pass
    r=requests.get(url,timeout=(5,12),headers={'User-Agent':'CryptoResearchCalendar/1.0'})
    r.raise_for_status()
    item={'fetched_at':now.isoformat(),'text':r.text}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(item),encoding='utf-8');temp.replace(path)
    return r.text,item['fetched_at']


def parse_ics(text,source):
    # BEA includes doubled CRs and RFC line folding.
    lines=[s.rstrip() for s in text.replace('\r','').split('\n') if s.strip()]
    unfolded=[]
    for line in lines:
        if line.startswith((' ','\t')) and unfolded:unfolded[-1]+=line[1:]
        else:unfolded.append(line)
    events=[]
    for block in '\n'.join(unfolded).split('BEGIN:VEVENT')[1:]:
        block=block.split('END:VEVENT',1)[0]
        if 'STATUS:CANCELLED' in block:continue
        summary=re.search(r'^SUMMARY(?:;[^:]*)?:(.*)$',block,re.M)
        start=re.search(r'^DTSTART([^:]*):(\d{8}T\d{6}Z?)$',block,re.M)
        if not summary or not start:continue
        title=summary[1].replace('\\,',',');kind=event_kind(title)
        if not kind:continue
        raw=start[2]
        if raw.endswith('Z'):stamp=pd.to_datetime(raw,format='%Y%m%dT%H%M%SZ',utc=True)
        else:
            tz=re.search(r'TZID=([^;]+)',start[1])
            if not tz:continue
            stamp=pd.to_datetime(raw,format='%Y%m%dT%H%M%S').tz_localize(tz[1]).tz_convert('UTC')
        events.append({'kind':kind,'title':title,'time_utc':stamp.isoformat(),'source':source,'provenance':'official'})
    return events


def parse_fed(text,source):
    headings=list(re.finditer(r'(\d{4}) FOMC Meetings',text));events=[]
    for i,m in enumerate(headings):
        year=int(m[1]);section=text[m.end():headings[i+1].start() if i+1<len(headings) else len(text)]
        for block in re.split(r'<div class="row fomc-meeting"',section)[1:]:
            month=re.search(r'fomc-meeting__month[^>]*>\s*<strong>(.*?)</strong>',block,re.S)
            date=re.search(r'fomc-meeting__date[^>]*>(.*?)</div>',block,re.S)
            if not month or not date:continue
            label=html.unescape(re.sub('<[^>]*>','',date[1])).strip()
            if re.search(r'unscheduled|notation|cancel',label,re.I):continue
            numbers=re.findall(r'\d+',label)
            if not numbers:continue
            name=html.unescape(month[1]).split('/')[-1].strip()
            try:stamp=pd.Timestamp(f'{name} {int(numbers[-1])}, {year} 14:00',tz='America/New_York').tz_convert('UTC')
            except ValueError:continue
            events.append({'kind':'FOMC','title':'Scheduled FOMC policy statement','time_utc':stamp.isoformat(),
                           'source':source,'provenance':'official_date_standard_time',
                           'time_basis':'14:00 America/New_York standard scheduled statement time; unscheduled events not covered'})
    return events


def parse_weekly(text,source,now):
    rows=json.loads(text)
    stamps=[utc(x['date']) for x in rows if x.get('date')]
    if not stamps:raise ValueError('Empty weekly calendar')
    first=min(stamps).tz_convert('America/New_York').normalize()
    sunday=first-pd.Timedelta(days=(first.dayofweek+1)%7)
    end=sunday+pd.DateOffset(days=7)
    covers=sunday.tz_convert('UTC')<=now and end.tz_convert('UTC')>=now+pd.Timedelta(hours=12)
    events=[]
    for row in rows:
        kind=event_kind(row.get('title',''))
        if row.get('country')=='USD' and kind:
            events.append({'kind':kind,'title':row['title'],'time_utc':utc(row['date']).isoformat(),
                           'source':source,'provenance':'third_party_calendar'})
    return events,covers


def fetch_calendar(now=None,cache_dir=None):
    now=utc(now);events=[];statuses={};coverage=set()
    def fetch(item):
        name,url=item
        try:
            text,stamp=cached_get(url,now,cache_dir)
            covers=False
            if name.startswith('ff_'):parsed,covers=parse_weekly(text,url,now)
            else:
                parsed=parse_fed(text,url) if name=='fed' else parse_ics(text,url)
                required={'bls':{'CPI','NFP'},'bea':{'PCE'},'fed':{'FOMC'}}[name]
                covers=all(any(e['kind']==kind and utc(e['time_utc'])>=now+pd.Timedelta(hours=12) for e in parsed) for kind in required)
            return name,parsed,{'status':'available' if covers else 'partial','fetched_at':stamp,'source':url,'parsed_events':len(parsed)},covers
        except Exception as exc:return name,[],{'status':'unavailable','source':url,'error_type':type(exc).__name__},False
    with ThreadPoolExecutor(max_workers=5) as pool:
        for name,parsed,status,covers in pool.map(fetch,SOURCES.items()):
            statuses[name]=status;events.extend(parsed)
            if covers:coverage.update({'bls':{'CPI','NFP'},'bea':{'PCE'},'fed':{'FOMC'}}.get(name,{'CPI','NFP','PCE','FOMC'}))
    # Retain source conflicts rather than silently choosing a convenient time.
    unique={(e['kind'],e['time_utc'],e['source']):e for e in events}
    upcoming=sorted([e for e in unique.values() if now<=utc(e['time_utc'])<=now+pd.Timedelta(hours=12)],key=lambda e:e['time_utc'])
    next_events=sorted([e for e in unique.values() if utc(e['time_utc'])>now],key=lambda e:e['time_utc'])[:12]
    return {'status':'available' if len(coverage)==4 else 'partial' if coverage else 'unavailable',
            'as_of':now.isoformat(),'coverage_hours':12 if len(coverage)==4 else 0,'covered_event_types':sorted(coverage),
            'events':upcoming,'next_events':next_events,'sources':statuses,
            'limitations':'Scheduled releases only. Third-party backup is labelled, not official verification; retain conflicting schedules. No unscheduled shock coverage.'}


def macro_veto(calendar,now=None):
    now=utc(now)
    try:
        age=(now-utc(calendar['as_of'])).total_seconds()
        if not 0<=age<=3600:return []
        return [e for e in calendar.get('events',[]) if e.get('kind') in {'FOMC','CPI','NFP','PCE'}
                and e.get('source') in SOURCES.values() and 0<=(utc(e['time_utc'])-now).total_seconds()<=4*3600]
    except (ValueError,KeyError,TypeError):return []
