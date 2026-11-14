from datetime import datetime

from shared.schema import Reading
from vitals.db.database import init_db, insert_reading, get_readings, get_latest

init_db()

insert_reading(Reading(astronaut_id="A001", timestamp=datetime.now(), vital="hr", value=72))
insert_reading(Reading(astronaut_id="A001", timestamp=datetime.now(), vital="spo2", value=98))

print("All readings:", get_readings("A001"))
print("Latest per vital:", get_latest("A001"))
print("DB test passed")