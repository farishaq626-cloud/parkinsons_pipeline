import json
from datetime import datetime
def save_report(data, filename="report.json"):
    report_content = {"run_date": datetime.now().strftime("%Y-%m-%d"), "metrics": data}
    with open(filename, "w") as f:
        json.dump(report_content, f, indent=4)