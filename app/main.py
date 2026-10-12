import os
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "attendance_db")]
db.employees.create_index([("emp_code", ASCENDING)], unique=True)
db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True)
db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)])

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
PRESENCE = {"PRESENT", "WFH", "ON_DUTY"}
STATUSES = PRESENCE | {"ABSENT", "LEAVE"}
app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


# Helpers
def fail_timestamp():
    raise HTTPException(status_code=422, detail="Timestamp must be epoch milliseconds")


def from_ms(value: int) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int) or value < 100_000_000_000:
        fail_timestamp()
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).replace(microsecond=0)
    except (ValueError, OverflowError, OSError):
        fail_timestamp()


def aware_utc(value):
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def to_ms(value):
    value = aware_utc(value)
    return int(value.timestamp() * 1000) if value else None


def rounded(value, places=2):
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def valid_date(value, field="date"):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"{field} must use YYYY-MM-DD")


def valid_month(value):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value or ""):
        raise HTTPException(status_code=422, detail="month must use YYYY-MM")
    year, month = map(int, value.split("-"))
    start = date(year, month, 1)
    end = date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    return start, end


def employee_or_404(emp_code):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp


def attendance_date(punch_in, emp):
    local = aware_utc(punch_in).astimezone(IST)
    day = local.date()
    if emp["shift_end"] <= emp["shift_start"] and local.strftime("%H:%M") < emp["shift_end"]:
        day -= timedelta(days=1)
    return day.isoformat()


def late_minutes(punch_in, shift_start):
    local = aware_utc(punch_in).astimezone(IST)
    h, m = map(int, shift_start.split(":"))
    start = local.replace(hour=h, minute=m, second=0, microsecond=0)
    minutes = int((local - start).total_seconds() // 60)
    return minutes if minutes > 10 else 0


def work_hours(punch_in, punch_out):
    seconds = (aware_utc(punch_out) - aware_utc(punch_in)).total_seconds()
    return rounded(Decimal(str(seconds)) / Decimal("3600"), 2)


def shift_end_datetime(day_string, shift_start, shift_end):
    h, m = map(int, shift_end.split(":"))
    end = datetime.combine(valid_date(day_string), time(h, m), tzinfo=IST)
    if shift_end <= shift_start:
        end += timedelta(days=1)
    return end


def overtime_minutes(punch_out, shift_start, shift_end, day_string):
    end = shift_end_datetime(day_string, shift_start, shift_end)
    minutes = int((aware_utc(punch_out).astimezone(IST) - end).total_seconds() // 60)
    return minutes if minutes >= 30 else 0


def normalize_attendance(doc):
    doc.pop("_id", None)
    for key in ("punch_in", "punch_out"):
        if isinstance(doc.get(key), datetime):
            doc[key] = to_ms(doc[key])
    for entry in doc.get("history", []):
        if isinstance(entry.get("at"), datetime):
            entry["at"] = to_ms(entry["at"])
    return doc


def working_days(start, end):
    total = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            total += 1
        current += timedelta(days=1)
    return total


def month_logs(month_start, month_end, emp_code=None):
    query = {"date": {"$gte": month_start.isoformat(), "$lte": month_end.isoformat()}}
    if emp_code:
        query["emp_code"] = emp_code
    return list(db.attendance_logs.find(query, {"_id": 0}))


# Models
class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("shift_start", "shift_end")
    @classmethod
    def check_shift_time(cls, value):
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("Shift time must use HH:MM")
        return value

    @field_validator("shift_end")
    @classmethod
    def check_shift_diff(cls, value, info):
        if info.data.get("shift_start") == value:
            raise ValueError("Shift start and end must differ")
        return value

    @field_validator("joined_on")
    @classmethod
    def check_joined_on(cls, value):
        valid_date(value, "joined_on")
        return value


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None


class RegularizeIn(BaseModel):
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# Health and employee endpoints
@app.get("/health")
def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC)
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")
    doc.pop("_id", None)
    doc["created_at"] = to_ms(doc["created_at"])
    return doc


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: int = 1, page_size: int = 20):
    if page < 1 or not 1 <= page_size <= 100:
        raise HTTPException(status_code=422, detail="Invalid pagination")
    query = {"department": department} if department else {}
    total = db.employees.count_documents(query)
    items = list(db.employees.find(query, {"_id": 0}).sort("emp_code", ASCENDING)
                 .skip((page - 1) * page_size).limit(page_size))
    for item in items:
        if isinstance(item.get("created_at"), datetime):
            item["created_at"] = to_ms(item["created_at"])
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# Attendance endpoints
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = employee_or_404(body.emp_code)
    if body.status not in PRESENCE:
        raise HTTPException(status_code=422, detail="Invalid attendance status")
    ts = from_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    doc = {
        "emp_code": body.emp_code, "date": attendance_date(ts, emp), "status": body.status,
        "punch_in": ts, "punch_out": None, "work_hours": None,
        "late_minutes": late_minutes(ts, emp["shift_start"]), "overtime_minutes": 0,
        "half_day": False, "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Already punched in for this date")
    doc.pop("_id", None)
    doc["punch_in"] = to_ms(ts)
    return doc


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = employee_or_404(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_out": None, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if not record:
        existing = db.attendance_logs.find_one(
            {"emp_code": body.emp_code}, sort=[("punch_in", DESCENDING)]
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="No attendance record found")
        if existing.get("punch_out") is not None:
            raise HTTPException(status_code=409, detail="Attendance record already closed")
        raise HTTPException(status_code=404, detail="No open punch-in record found")

    pin = aware_utc(record["punch_in"])
    seconds = (ts - pin).total_seconds()
    if seconds <= 0 or seconds > 86400:
        raise HTTPException(status_code=422, detail="Invalid punch-out time")
    hours = work_hours(pin, ts)
    half = hours < 4.50
    overtime = overtime_minutes(ts, emp["shift_start"], emp["shift_end"], record["date"])
    result = db.attendance_logs.update_one(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": ts, "work_hours": hours, "half_day": half,
                  "overtime_minutes": overtime}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=409, detail="Record already punched out")
    return {
        "emp_code": body.emp_code, "date": record["date"], "status": record["status"],
        "punch_in": to_ms(pin), "punch_out": to_ms(ts), "work_hours": hours,
        "late_minutes": record["late_minutes"], "overtime_minutes": overtime,
        "half_day": half, "history": record.get("history", []),
    }


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None, date_from: Optional[str] = None,
    date_to: Optional[str] = None, status: Optional[str] = None,
    page: int = 1, page_size: int = 20,
):
    if page < 1 or not 1 <= page_size <= 100:
        raise HTTPException(status_code=422, detail="Invalid pagination")
    if date_from:
        valid_date(date_from, "date_from")
    if date_to:
        valid_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to")
    if status and status not in STATUSES:
        raise HTTPException(status_code=422, detail="Invalid status")
    query = {}
    if emp_code:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from
        if date_to:
            query["date"]["$lte"] = date_to
    if status:
        query["status"] = status
    total = db.attendance_logs.count_documents(query)
    docs = list(db.attendance_logs.find(query, {"_id": 0})
                .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
                .skip((page - 1) * page_size).limit(page_size))
    return {"items": [normalize_attendance(x) for x in docs], "total": total,
            "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    valid_date(date)
    emp = employee_or_404(emp_code)
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    status = body.status if body.status is not None else record["status"]
    if status not in STATUSES:
        raise HTTPException(status_code=422, detail="Invalid attendance status")
    if status in {"ABSENT", "LEAVE"} and (body.punch_in is not None or body.punch_out is not None):
        raise HTTPException(status_code=422, detail="ABSENT/LEAVE cannot have punch times")

    pin = from_ms(body.punch_in) if body.punch_in is not None else aware_utc(record.get("punch_in"))
    pout = from_ms(body.punch_out) if body.punch_out is not None else aware_utc(record.get("punch_out"))
    if status in {"ABSENT", "LEAVE"}:
        pin = pout = None
    elif pin is None:
        raise HTTPException(status_code=422, detail="Presence status requires punch_in")

    if pin and attendance_date(pin, emp) != date:
        raise HTTPException(status_code=422, detail="punch_in does not match attendance date")
    if pout:
        if not pin:
            raise HTTPException(status_code=422, detail="punch_out requires punch_in")
        seconds = (pout - pin).total_seconds()
        if seconds <= 0 or seconds > 86400:
            raise HTTPException(status_code=422, detail="Invalid punch_out time")

    late = late_minutes(pin, emp["shift_start"]) if pin else 0
    hours = work_hours(pin, pout) if pin and pout else None
    half = hours < 4.50 if hours is not None else False
    overtime = overtime_minutes(pout, emp["shift_start"], emp["shift_end"], date) if pout else 0
    values = {"status": status, "punch_in": pin, "punch_out": pout, "work_hours": hours,
              "late_minutes": late, "overtime_minutes": overtime, "half_day": half}

    changes = {}
    for key, new in values.items():
        old = record.get(key)
        if key in {"punch_in", "punch_out"}:
            old = aware_utc(old) if old else None
        if old != new:
            changes[key] = ({"from": to_ms(old), "to": to_ms(new)}
                            if key in {"punch_in", "punch_out"} else {"from": old, "to": new})
    if not changes:
        raise HTTPException(status_code=422, detail="No changes supplied")

    history_entry = {"at": datetime.now(UTC), "by": body.regularized_by,
                     "reason": body.reason, "changes": changes}
    result = db.attendance_logs.update_one(
        {"_id": record["_id"], "history": record.get("history", [])},
        {"$set": values, "$push": {"history": history_entry}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=409, detail="Record changed concurrently")
    return normalize_attendance(db.attendance_logs.find_one({"_id": record["_id"]}))


# Analytics
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str):
    start, end = valid_month(month)
    emp = employee_or_404(emp_code)
    joined = valid_date(emp["joined_on"], "joined_on")
    effective_start = max(start, joined)
    wd = working_days(effective_start, end) if effective_start <= end else 0
    logs = month_logs(start, end, emp_code)
    present = leaves = late_count = 0.0
    hours = []
    for log in logs:
        d = valid_date(log["date"])
        if d.weekday() < 5 and d >= joined:
            if log["status"] in PRESENCE:
                present += 0.5 if log.get("half_day") else 1
            elif log["status"] == "LEAVE":
                leaves += 1
        if log.get("late_minutes", 0) > 0:
            late_count += 1
        if log["status"] in PRESENCE and log.get("work_hours") is not None:
            hours.append(log["work_hours"])
    pct = rounded(Decimal(str(present)) * 100 / Decimal(str(wd)), 2) if wd else None
    return {"emp_code": emp_code, "month": month, "working_days": wd,
            "present_days": rounded(present), "leave_days": int(leaves),
            "late_count": int(late_count), "attendance_pct": pct,
            "avg_work_hours": rounded(sum(hours) / len(hours)) if hours else None}


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(...), department: Optional[str] = None):
    start, end = valid_month(month)
    employees = list(db.employees.find({"department": department} if department else {}))
    emp_map = {e["emp_code"]: e for e in employees}
    rows = {}
    for emp in employees:
        if valid_date(emp["joined_on"]) <= end:
            rows.setdefault(emp["department"], {"department": emp["department"], "headcount": 0,
                "present_days": 0.0, "late_count": 0, "total_late_minutes": 0,
                "leave_count": 0, "on_duty_count": 0, "_hours": []})
            rows[emp["department"]]["headcount"] += 1
    for log in month_logs(start, end):
        emp = emp_map.get(log["emp_code"])
        if not emp or valid_date(emp["joined_on"]) > end:
            continue
        row = rows.get(emp["department"])
        if row is None:
            continue
        d = valid_date(log["date"])
        if d.weekday() < 5 and d >= valid_date(emp["joined_on"]) and log["status"] in PRESENCE:
            row["present_days"] += 0.5 if log.get("half_day") else 1
        if log.get("late_minutes", 0) > 0:
            row["late_count"] += 1
            row["total_late_minutes"] += log["late_minutes"]
        if log["status"] == "LEAVE":
            row["leave_count"] += 1
        if log["status"] == "ON_DUTY":
            row["on_duty_count"] += 1
        if log["status"] in PRESENCE and log.get("work_hours") is not None:
            row["_hours"].append(log["work_hours"])
    items = []
    for name in sorted(rows):
        row = rows[name]
        if row["headcount"] == 0:
            continue
        hours = row.pop("_hours")
        row["present_days"] = rounded(row["present_days"])
        row["avg_work_hours"] = rounded(sum(hours) / len(hours)) if hours else None
        items.append(row)
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: str, limit: int = 10, department: Optional[str] = None):
    start, end = valid_month(month)
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    employees = {e["emp_code"]: e for e in db.employees.find({})}
    totals = {}
    for log in month_logs(start, end):
        emp = employees.get(log["emp_code"])
        minutes = log.get("late_minutes", 0)
        if not emp or minutes <= 0 or (department and emp["department"] != department):
            continue
        row = totals.setdefault(log["emp_code"], {"emp_code": log["emp_code"], "name": emp["name"],
            "department": emp["department"], "total_late_minutes": 0, "late_count": 0})
        row["total_late_minutes"] += minutes
        row["late_count"] += 1
    ordered = sorted(totals.values(), key=lambda x: (-x["total_late_minutes"], x["emp_code"]))
    previous = None
    rank = 0
    for index, row in enumerate(ordered, 1):
        if row["total_late_minutes"] != previous:
            rank = index
            previous = row["total_late_minutes"]
        row["rank"] = rank
    if len(ordered) > limit:
        cutoff = ordered[limit - 1]["total_late_minutes"]
        ordered = [x for x in ordered if x["total_late_minutes"] >= cutoff]
    return {"month": month, "items": ordered}


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, date_from: str, date_to: str):
    start, end = valid_date(date_from, "date_from"), valid_date(date_to, "date_to")
    if end < start or (end - start).days + 1 > 92:
        raise HTTPException(status_code=422, detail="Invalid date range")
    employees = list(db.employees.find({"department": department}, {"_id": 0}))
    if not employees:
        raise HTTPException(status_code=404, detail="Department not found")
    emp_map = {e["emp_code"]: e for e in employees}
    logs = list(db.attendance_logs.find({"emp_code": {"$in": list(emp_map)},
        "date": {"$gte": date_from, "$lte": date_to}}, {"_id": 0}))
    by_day = {}
    for log in logs:
        by_day.setdefault(log["date"], []).append(log)
    items = []
    day = start
    while day <= end:
        active = [e for e in employees if valid_date(e["joined_on"]) <= day]
        present = 0.0
        late = 0
        for log in by_day.get(day.isoformat(), []):
            emp = emp_map[log["emp_code"]]
            if valid_date(emp["joined_on"]) > day:
                continue
            if log["status"] in PRESENCE:
                present += 0.5 if log.get("half_day") else 1
            if log.get("late_minutes", 0) > 0:
                late += 1
        rate = (rounded(Decimal(str(present)) * 100 / Decimal(len(active)), 4)
                if day.weekday() < 5 and active else (0.0 if day.weekday() < 5 else None))
        items.append({"date": day.isoformat(), "headcount": len(active),
                      "present_count": rounded(present), "late_count": late,
                      "attendance_rate": rate})
        day += timedelta(days=1)
    for i, row in enumerate(items):
        recent = [x["attendance_rate"] for x in items[max(0, i - 6):i + 1]
                  if x["attendance_rate"] is not None]
        row["moving_avg_7d"] = rounded(sum(recent) / len(recent), 4) if recent else None
    return {"department": department, "date_from": date_from, "date_to": date_to, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str, emp_code: Optional[str] = None, month: Optional[str] = None,
    department: Optional[str] = None, date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    if endpoint == "attendance_list":
        query = {}
        if emp_code:
            query["emp_code"] = emp_code
        if date_from or date_to:
            query["date"] = {}
            if date_from:
                valid_date(date_from, "date_from")
                query["date"]["$gte"] = date_from
            if date_to:
                valid_date(date_to, "date_to")
                query["date"]["$lte"] = date_to
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="Invalid date range")
        command = {"find": "attendance_logs", "filter": query,
                   "sort": {"date": -1, "emp_code": 1}}
    elif endpoint in {"employee_monthly", "department_summary", "late_leaderboard"}:
        if not month:
            raise HTTPException(status_code=422, detail="month is required")
        start, end = valid_month(month)
        query = {"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
        if endpoint == "employee_monthly":
            if not emp_code:
                raise HTTPException(status_code=422, detail="emp_code is required")
            query["emp_code"] = emp_code
        elif endpoint == "department_summary" and department:
            codes = [x["emp_code"] for x in db.employees.find({"department": department}, {"emp_code": 1})]
            query["emp_code"] = {"$in": codes}
        elif endpoint == "late_leaderboard":
            query["late_minutes"] = {"$gt": 0}
        command = {"find": "attendance_logs", "filter": query}
    elif endpoint == "department_trend":
        if not department or not date_from or not date_to:
            raise HTTPException(status_code=422, detail="department, date_from and date_to are required")
        start, end = valid_date(date_from, "date_from"), valid_date(date_to, "date_to")
        if end < start or (end - start).days + 1 > 92:
            raise HTTPException(status_code=422, detail="Invalid date range")
        codes = [x["emp_code"] for x in db.employees.find({"department": department}, {"emp_code": 1})]
        command = {"find": "attendance_logs", "filter": {
            "emp_code": {"$in": codes}, "date": {"$gte": date_from, "$lte": date_to}}}
    else:
        raise HTTPException(status_code=422, detail="Unknown endpoint")
    return db.command("explain", command, verbosity="executionStats")
