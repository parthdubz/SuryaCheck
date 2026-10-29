from django.db import models


class Plant(models.Model):
    plant_id = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=64)

    def __str__(self):
        return self.name


class Inverter(models.Model):
    plant = models.ForeignKey(Plant, on_delete=models.CASCADE, related_name="inverters")
    source_key = models.CharField(max_length=64)
    inverter_id = models.CharField(max_length=96, unique=True)
    display_number = models.PositiveIntegerField(default=1)
    peak_power_kw = models.FloatField(null=True, blank=True)
    is_unseen_in_training = models.BooleanField(default=False)

    @property
    def short_name(self):
        return f"Inverter {self.display_number:02d}"

    @property
    def display_name(self):
        return f"{self.plant.name} - {self.short_name}"

    def __str__(self):
        return self.inverter_id


class Reading(models.Model):
    inverter = models.ForeignKey(Inverter, on_delete=models.CASCADE, related_name="readings")
    timestamp = models.DateTimeField()
    day_index = models.IntegerField()
    ac_power = models.FloatField()
    dc_power = models.FloatField(null=True)
    irradiation = models.FloatField()
    module_temperature = models.FloatField()
    ambient_temperature = models.FloatField()
    is_synthetic_fault = models.BooleanField(default=False)

    class Meta:
        indexes = [models.Index(fields=["inverter", "timestamp"])]


class Prediction(models.Model):
    inverter = models.ForeignKey(Inverter, on_delete=models.CASCADE, related_name="predictions")
    timestamp = models.DateTimeField()
    model_name = models.CharField(max_length=32)
    predicted_ac_power = models.FloatField()

    class Meta:
        indexes = [models.Index(fields=["inverter", "timestamp", "model_name"])]


class Alert(models.Model):
    STATUS_CHOICES = [("open", "Open"), ("dismissed", "Dismissed")]
    inverter = models.ForeignKey(Inverter, on_delete=models.CASCADE, related_name="alerts")
    fault_type = models.CharField(max_length=32)
    start_day = models.IntegerField()
    detected_day = models.IntegerField(null=True, blank=True)
    ratio = models.FloatField()
    rupees_lost = models.FloatField(default=0.0)
    energy_deficit_kwh = models.FloatField(default=0.0)
    likely_cause = models.CharField(max_length=32, blank=True, default="")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="open")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.inverter.inverter_id} {self.fault_type} day {self.start_day}"


class Feedback(models.Model):
    alert = models.ForeignKey(Alert, on_delete=models.CASCADE, related_name="feedback")
    dismissed = models.BooleanField(default=True)
    reason = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ModelRun(models.Model):
    name = models.CharField(max_length=64)
    metrics_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} @ {self.created_at}"
