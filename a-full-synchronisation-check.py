"""
Might need to tick "Delete remote files when local are deleted" in
Tools -> Deployment -> Options
"""

import os
import datetime

print("This line was written on a laptop last week.")
print(os.getcwd())

print(f"While this one was written on {datetime.date.today()}")
