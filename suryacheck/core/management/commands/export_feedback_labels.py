import csv

from django.conf import settings
from django.core.management.base import BaseCommand

from core.models import Alert


class Command(BaseCommand):
    help = (
        "Export every Alert and its installer Feedback as a labeled CSV. "
        "This is the project's answer to 'there are no real fault labels': "
        "each dismissal an installer makes (confirmed fault, or false alarm "
        "with a reason) is a real, human-verified label that can retrain the "
        "detector and classifier in the future, instead of relying only on "
        "synthetic faults forever."
    )

    def handle(self, *args, **options):
        settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = settings.ARTIFACTS_DIR / "feedback_labels.csv"

        alerts = Alert.objects.select_related("inverter", "inverter__plant").prefetch_related("feedback")

        rows = []
        for alert in alerts:
            feedback_entries = list(alert.feedback.all())
            if feedback_entries:
                for fb in feedback_entries:
                    rows.append(self._row(alert, fb))
            else:
                rows.append(self._row(alert, None))

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "inverter_id", "plant_id", "fault_type_predicted", "start_day",
                "detected_day", "ratio", "energy_deficit_kwh", "rupees_lost",
                "alert_status", "human_dismissed", "human_reason", "feedback_created_at",
            ])
            writer.writerows(rows)

        self.stdout.write(self.style.SUCCESS(
            f"Wrote {len(rows)} labeled rows to {out_path} "
            f"({sum(1 for r in rows if r[9] is not None)} with real installer feedback)."
        ))

    def _row(self, alert, feedback):
        return [
            alert.inverter.inverter_id,
            alert.inverter.plant.plant_id,
            alert.fault_type,
            alert.start_day,
            alert.detected_day,
            alert.ratio,
            alert.energy_deficit_kwh,
            alert.rupees_lost,
            alert.status,
            feedback.dismissed if feedback else None,
            feedback.reason if feedback else "",
            feedback.created_at.isoformat() if feedback else "",
        ]
